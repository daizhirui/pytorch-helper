import random
from abc import ABC
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn.modules as nn
from torch import distributed
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from pytorch_helper.launcher.base import LauncherTask
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.utils import log
from pytorch_helper.utils.dist import get_rank
from pytorch_helper.utils.dist import is_distributed
from pytorch_helper.utils.dist import reduce_value

__all__ = [
    'TaskBase',
    'BatchPack'
]


@dataclass
class BatchPack:
    gt: Any
    batch_size: int = None
    pred: Any = None
    loss: Any = None

    @property
    def batch(self):
        return self.gt


class TaskBase(LauncherTask, ABC):
    STAGE_TRAIN = 'train'
    STAGE_VALID = 'valid'
    STAGE_TEST = 'test'

    def __init__(self, task_option: TaskOption):
        """ TaskBase finishes the following initialization steps:
        - random seed setting
        - data parallel checking and setting

        :param task_option: task option instance
        """
        super(TaskBase, self).__init__()
        self.init_random_seed()
        self._option = task_option

        # gpu setting
        gpu_ids = self._option.gpu_ids
        self.is_distributed = is_distributed()
        # check data parallel: if distributed data parallel is enabled, should
        # check visible gpu devices via torch.cuda.device_count(), otherwise,
        # check length of gpu_ids, will use data parallel instead.
        if self.is_distributed:
            n_gpus = torch.cuda.device_count()
        else:
            n_gpus = len(gpu_ids)
        self.is_parallel = n_gpus > 1 or self.is_distributed

        # distributed data parallel, data parallel or single gpu
        self.rank = get_rank()
        self.is_rank0 = self.rank == 0
        # setup device
        torch.cuda.set_device(gpu_ids[0])

        # configure tasks option
        if self.is_distributed:  # DistributedDataParallel (DDP)
            n_gpus = distributed.get_world_size()
            batch_size = self.option.dataloader.kwargs['batch_size']
            self.option.dataloader.kwargs['batch_size'] //= n_gpus
            self.option.dataloader.kwargs['num_workers'] //= n_gpus
            rest = batch_size % n_gpus
            if rest > 0:
                raise IOError(
                    f'Unbalanced batch size distribution: batch size '
                    f'is {self.option.dataloader.kwargs["batch_size"]} '
                    f'for {n_gpus} GPUs.'
                )

        log.info(__name__, f'Start task: {self.option.name}')
        log.info(__name__, f'Datetime: {self.option.datetime}')

        if self.is_parallel:
            if self.is_distributed:
                log.info(
                    __name__, f'DDP Process {distributed.get_rank()} online'
                )
            else:
                log.info(
                    __name__, f'Using {n_gpus} GPUs: {gpu_ids} for DataParallel'
                )
                batch_size = self.option.dataloader.kwargs['batch_size']
                if batch_size % n_gpus != 0:
                    log.warn(
                        __name__,
                        f'batch size {batch_size} cannot be distributed onto '
                        f'{n_gpus} GPUs evenly!'
                    )

        # model
        self.model, state_dict = self.option.model.build()
        self.model.cuda()
        # unwrapped model
        if self.is_parallel:
            if self.is_distributed:
                log.info(
                    __name__, 'Using DDP, convert BatchNorm to SyncBatchNorm'
                )
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
        self.load_state(state_dict, 'loss_fn', self.loss_fn)
        # dataloader
        self.dataloader = self.option.dataloader.build()
        self.cur_dataloader = None
        # optimizer
        self.optimizer = None
        self.lr_scheduler = None
        # miscellaneous
        self.epoch = self.option.train_setting.start_epoch
        self.current_train_routine = None
        self.cur_stage = None
        self.tboard = None
        # post init by descendents
        self.post_init(state_dict)

    @property
    def option(self):
        return self._option

    def post_init(self, state_dict: dict):
        """ Do some initialization related to state_dict

        :param state_dict: a dict of task state
        """
        raise NotImplementedError

    @staticmethod
    def load_state(state_dict: dict, key: str, obj):
        """ load the state of `obj` from `state_dict[key]`

        :param state_dict: dict of object stats
        :param key: reference to find the state for `obj`
        :param obj: the object to load state
        """
        if state_dict is None or obj is None:
            return
        if key in state_dict:
            state_dict = state_dict[key]
        if state_dict is None:
            return
        if len(state_dict) > 0:
            try:
                obj.load_state_dict(state_dict)
            except Exception as e:
                log.warn(__name__, repr(e))

    @staticmethod
    def get_state(obj) -> Optional[dict]:
        """ get the state dict of `obj`

        :param obj: the object to get its state dict
        :return:
        """
        if obj and hasattr(obj, 'state_dict'):
            return obj.state_dict()

    def state_dict(self) -> dict:
        """ get the dict of task state: task option, model state, loss function
        state, current epoch, and random generator states.

        :return: dict of task state
        """
        return dict(
            option=self.option.asdict(),
            model=self.get_state(self.unwrapped_model),
            loss_fn=self.get_state(self.loss_fn),
            epoch=self.epoch,
            rng_state=self.get_rng_state()
        )

    # model
    def model_forward_backward(
            self, batch_pack: BatchPack, backward=False
    ) -> BatchPack:
        """ This method should define how to perform model forward and
        backward propagation, and update `batch_pack.pred`, `batch_pack.loss`,
        `batch_pack.batch_size`. To make the loss synchronized across gpus,
        call `self.sync_value`.

        :param batch_pack: BatchPack that stores ground truth, prediction, loss
            and batch size
        :param backward: Bool to indicate whether to perform backward
            propagation
        :return: batch, model output, loss and batch_size
        """
        raise NotImplementedError

    def freeze_and_unfreeze_modules(
            self, names: Union[str, Sequence[str]], reset_optimizer: bool = True
    ):
        """ this method freezes the model and then unfreezes modules specified
        by `names`

        :param names: Sequence of module names, should be seekable via `getattr`
        :param reset_optimizer: Bool to reset the optimizer such that only
            unfrozen modules will be updated by the optimizer, default True
        """
        self.unwrapped_model.train(False)
        for parameter in self.unwrapped_model.parameters():
            parameter.requires_grad = False
        for name in names:
            log.info(__name__, f"train {name}")
            name = name.split('.')
            module = self.unwrapped_model
            for n in name:
                module = getattr(module, n)
            module.train(True)
            for parameter in module.parameters():
                parameter.requires_grad = True
        if reset_optimizer:
            if not hasattr(self, 'optimizer'):
                log.warn(__name__, 'no optimizer to reset')
                return

            lr = self.learning_rate
            self.optimizer = self.option.optimizer.build(self.unwrapped_model)
            self.set_learning_rate(lr)
            log.info(__name__, 'optimizer reset')

            if not hasattr(self, 'lr_scheduler'):
                return
            self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)
            log.info(__name__, 'lr scheduler reset')

    @staticmethod
    def sync_value(input_value):
        """ automatically synchronize the result across different gpus. This is
        safe to use in any conditions including single-gpu cases, which will
        return the `input_value` directly.

        :param input_value: value or dict of input value
        :return: reduced value or dict of reduced values
        """
        return reduce_value(input_value)

    # learning rate
    @property
    def learning_rate(self):
        """ get the learning rate

        :return: double of learning rate
        """
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        """ set the learning rate

        :param lr: double of learning rate
        """
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    # random generator state
    @staticmethod
    def init_random_seed(seed: int = 0):
        """ set the initial random seed of torch, torch.cuda, numpy, and random

        :param seed: int
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def get_rng_state():
        """ get the states of all the random generators

        :return: dict of random generator states
        """
        seed = dict(
            numpy=np.random.get_state(),
            random=random.getstate(),
            torch=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all()
        )
        return seed

    @staticmethod
    def set_rng_state(rng_state: dict):
        """ set the states of all the random generators

        :param rng_state: dict of random generator states
        """
        set_state_fns = dict(
            numpy=np.random.set_state,
            random=random.setstate,
            torch=torch.set_rng_state,
            torch_cuda=torch.cuda.set_rng_state_all
        )
        for key, set_state_fn in set_state_fns.items():
            if key in rng_state:
                set_state_fn(rng_state[key])
            else:
                log.warn(
                    __name__, f'random state for {key} is missing!'
                )
