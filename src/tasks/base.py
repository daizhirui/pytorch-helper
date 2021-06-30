import numpy as np
import torch
import torch.nn.modules as nn
from torch import distributed
from torch.distributed import all_reduce
from torch.distributed import get_world_size
from torch.distributed import is_initialized as distributed_initialized
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from utils import log

__all__ = ['TaskBase']


class TaskBase:
    """
    TaskBase finishes all the initialization tasks:
    - random seed setting
    - data parallel checking and setting
    """

    def __init__(self, task_option, gpu_ids):
        self.init_random_seed()
        self.gpu_ids = gpu_ids
        # the library supports distributed data parallel
        self.is_distributed = hasattr(distributed, 'is_initialized')
        # distributed data parallel is initialized
        self.is_distributed = self.is_distributed and distributed_initialized()
        # check data parallel: if distributed data parallel is enabled, should
        # check visible gpu devices via torch.cuda.device_count(), otherwise,
        # check length of gpu_ids, will use data parallel instead.
        if self.is_distributed:
            n_gpus = torch.cuda.device_count()
        else:
            n_gpus = len(gpu_ids)
        self.is_parallel = n_gpus > 1 or self.is_distributed

        # distributed data parallel, data parallel or single gpu
        self.rank = distributed.get_rank() if self.is_distributed else 0
        self.is_rank0 = self.rank == 0
        # setup device
        torch.cuda.set_device(gpu_ids[0])

        # configure tasks option
        from settings.options.task import TaskOption
        self.option: TaskOption = task_option
        if self.is_distributed:  # DistributedDataParallel (DDP)
            n_gpus = distributed.get_world_size()
            batch_size = self.option.dataloader.kwargs['batch_size']
            self.option.dataloader.kwargs['batch_size'] //= n_gpus
            self.option.dataloader.kwargs['num_workers'] //= n_gpus
            rest = batch_size % n_gpus
            if rest > 0 and 0 < gpu_ids[0] <= rest:
                self.option.dataloader.kwargs['batch_size'] += 1
                log.warn(f"Unbalanced batch size distribution: "
                         f"{self.option.dataloader.kwargs['batch_size']}")

        log.info(f'Start task: {self.option.name}')
        log.info(f'Datetime: {self.option.datetime}')

        if self.is_parallel:
            if self.is_distributed:
                log.info(f'DDP Process {distributed.get_rank()} online')
            else:
                log.info(f'Using {n_gpus} GPUs: {gpu_ids} for DataParallel')
                batch_size = self.option.dataloader.kwargs['batch_size']
                if batch_size % n_gpus != 0:
                    log.warn(
                        f'batch size {batch_size} cannot be distributed onto '
                        f'{n_gpus} GPUs evenly!'
                    )

        # model
        self.model, state_dict = self.option.model.build()
        self.model.cuda()
        # unwrapped model
        if self.is_parallel:
            if self.is_distributed:
                log.info('Using DDP, convert BatchNorm to SyncBatchNorm')
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = DistributedDataParallel(
                    module=self.model,
                    device_ids=gpu_ids,
                    output_device=gpu_ids[0],
                    find_unused_parameters=True
                )
                # `find_unused_parameters=True` allows the model to be
                # partially updated
            else:
                self.model = DataParallel(
                    module=self.model,
                    device_ids=gpu_ids,
                    output_device=gpu_ids[0]
                )
            self.unwrapped_model = self.model.module
        else:
            self.unwrapped_model = self.model
        # loss
        self.loss_fn = self.option.loss.build()
        self.loss_min = None
        if state_dict and 'loss_fn' in state_dict:
            if len(state_dict['loss_fn']) > 0:
                self.loss_fn.load_state(state_dict['loss_fn'])
        # dataloader
        self.dataloader = self.option.dataloader.build()
        # optimizer
        self.optimizer = None
        self.lr_scheduler = None
        # miscellaneous
        self.epoch = self.option.train_setting.start_epoch
        self.current_train_routine = None
        # post init by descendents
        self._post_init(state_dict)

    def _post_init(self, state_dict):
        raise NotImplementedError

    def model_forward_backward(self, batch, backward=False):
        raise NotImplementedError

    def sync_value(self, value):
        if self.is_distributed:
            # get the loss average across the whole world,
            # i.e. over the whole batch
            def single_sync(_value):
                if _value is None:
                    return _value
                all_reduce(_value.data, op=distributed.ReduceOp.SUM)
                size = float(get_world_size())
                _value.data /= size
                return _value

            if isinstance(value, dict):
                for key, l in value.items():
                    value[key] = single_sync(l)
            else:
                value = single_sync(value)
        return value

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def freeze_and_unfreeze_modules(self, names, reset_optimizer=False):
        self.unwrapped_model.train(False)
        for parameter in self.unwrapped_model.parameters():
            parameter.requires_grad = False
        for name in names:
            log.info(f"train {name}")
            name = name.split('.')
            module = self.unwrapped_model
            for n in name:
                module = getattr(module, n)
            module.train(True)
            for parameter in module.parameters():
                parameter.requires_grad = True
        if reset_optimizer:
            if not hasattr(self, 'optimizer'):
                log.info("no optimizer to reset")
                return

            lr = self.learning_rate
            self.optimizer = self.option.optimizer.build(self.unwrapped_model)
            self.set_learning_rate(lr)
            log.info("optimizer reset")

            if not hasattr(self, 'lr_scheduler'):
                return
            self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)
            log.info("lr scheduler reset")

    @staticmethod
    def init_random_seed(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_rng_state():
        seed = dict(
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all()
        )
        return seed

    @staticmethod
    def set_rng_state(rng_state):
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
