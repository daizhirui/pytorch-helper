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
from pytorch_helper.utils.dist import get_rank
from pytorch_helper.utils.dist import is_distributed
from pytorch_helper.utils.dist import reduce_value
from pytorch_helper.utils.log import get_logger

__all__ = [
    'TaskBase',
    'Batch'
]

logger = get_logger(__name__)


@dataclass
class Batch:
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
        gpu_ids = self._option.cuda_ids
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

        logger.info(f'Start task: {self.option.name}')
        logger.info(f'Datetime: {self.option.datetime}')

        if self.is_parallel:
            if self.is_distributed:
                logger.info(f'DDP Process {distributed.get_rank()} online')
            else:
                logger.info(f'Using {n_gpus} GPUs: {gpu_ids} for DataParallel')
                batch_size = self.option.dataloader.kwargs['batch_size']
                if batch_size % n_gpus != 0:
                    logger.warn(
                        f'batch size {batch_size} cannot be distributed onto '
                        f'{n_gpus} GPUs evenly!'
                    )

        # model
        self.model, state_dict = self.option.model.build()
        self.model.cuda()
        # unwrapped model
        if self.is_parallel:
            if self.is_distributed:
                logger.info('Using DDP, convert BatchNorm to SyncBatchNorm')
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
        self.batch_cnt = {
            self.STAGE_TRAIN: 0,
            self.STAGE_VALID: 0,
            self.STAGE_TEST: 0,
            'all': 0
        }
        # post init by descendants
        self.post_init(state_dict)

    # model
    def model_forward_backward(
        self, batch: Batch, backward: bool = False
    ) -> Batch:
        """ This method should define how to perform model forward and
        backward propagation, and update `batch_pack.pred`, `batch_pack.loss`,
        `batch_pack.batch_size`. To make the loss synchronized across gpus,
        call `self.sync_value`.

        :param batch: BatchPack that stores ground truth, prediction, loss
            and batch size
        :param backward: Bool to indicate whether to perform backward
            propagation
        :return: batch, model output, loss and batch_size
        """
        raise NotImplementedError

    @staticmethod
    def sync_value(input_value):
        """ automatically synchronize the result across different gpus. This is
        safe to use in any conditions including single-gpu cases, which will
        return the `input_value` directly.

        :param input_value: value or dict of input value
        :return: reduced value or dict of reduced values
        """
        return reduce_value(input_value)
