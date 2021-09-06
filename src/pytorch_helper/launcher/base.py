import os
from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Type

from .parse import MainArg
from ..settings.options.task import TaskOption
from ..utils.dist import is_distributed
from ..utils.dist import is_rank0
from ..utils.dist import synchronize
from ..utils.io import load_yaml
from ..utils.log import get_logger

__all__ = [
    'LauncherTask',
    'Launcher',
    'run_task'
]

logger = get_logger(__name__)


class LauncherTask:
    """ Base class of Tasks used by Launcher
    """

    def run(self):
        """ run the task
        """
        raise NotImplementedError

    def backup(self, immediate: bool, resumable: bool):
        """ backup the task status

        :param immediate: Bool to backup immediately
        :param resumable: Bool to backup states for resuming the task
        """
        raise NotImplementedError

    @property
    def option(self):
        """ return the task option
        """
        raise NotImplementedError


def run_task(
    cuda_ids: List[int], main_args: MainArg, task_option: TaskOption,
    register_func: Callable
):
    """ default function used to run the task

    :param cuda_ids: Sequence of CUDA device indices
    :param main_args: Dict of arguments parsed from the command line
    :param task_option: TaskOption used to build the task
    :param register_func: Callable to setup `settings.space.Spaces`, used
            before building the task
    """
    if main_args.wait_gpus:
        from ..utils.gpu.wait_gpus import wait_gpus
        wait_gpus(OrderedDict(
            (x, main_args.cuda_device_mapping[x]) for x in cuda_ids
        ))
    synchronize()

    register_func()
    from pytorch_helper.settings.space import Spaces
    task_option.cuda_ids = cuda_ids
    task: LauncherTask = Spaces.build_task(task_option)
    try:
        task.run()
    except Exception as e:
        raise e
    finally:
        if is_rank0() and task.option.train:
            logger.info('backup the task')
            task.backup(immediate=True, resumable=True)
        if is_distributed():
            from torch.distributed import destroy_process_group
            destroy_process_group()


class Launcher:
    def __init__(
        self, arg_cls: Type[MainArg], register_func: Callable,
        for_train: bool
    ):
        """ Base class of launchers for building and running a task properly

        :param arg_cls:
        :param register_func: Callable to setup `settings.space.Spaces`, used
            before building the task
        :param for_train:
        """
        # pytorch cannot be imported before this line
        self.args = arg_cls.parse()
        self.register_func = register_func
        self.for_train = for_train

        from torch import distributed
        self.is_distributed = len(self.args.use_gpus) > 1 \
                              and distributed.is_available() \
                              and not self.args.use_data_parallel

        task_dict = load_yaml(self.args.task_option_file)
        task_dict = self.modify_task_dict(task_dict)

        self.register_func()

        from pytorch_helper.settings.space import Spaces
        self.task_option = Spaces.build_task_option(task_dict)
        print(self.task_option)

    def modify_task_dict(self, task_dict: dict) -> dict:
        """ modify the task option dict before building the task option

        :param task_dict: Dict used to build the task option
        :return: modified task_dict
        """
        task_dict['for_train'] = self.for_train
        task_dict['resume'] = self.args.resume
        if self.args.pth_path:
            task_dict['model']['pth_path'] = self.args.pth_path
        if self.args.dataset_path:
            task_dict['dataset_path'] = self.args.dataset_path
        if self.args.output_path:
            task_dict['output_path'] = self.args.output_path
        task_dict['is_distributed'] = self.is_distributed

        task_dict['test_option'] = None
        if self.args.test_option_file:
            if not os.path.isfile(self.args.test_option_file):
                raise FileNotFoundError(
                    f'{self.args.test_option_file} does not exist or '
                    f'is not a file'
                )
            task_dict['test_option'] = load_yaml(self.args.test_option_file)

        return task_dict

    def run(self, run_task_func: Callable = None, *run_task_func_args):
        """ run the task by `run_task_func` in a proper method: single-gpu,
        multi-gpu DataParallel or multi-gpu DistributedDataParallel

        :param run_task_func: Callable to run the task
        :param run_task_func_args: arguments for `run_task_func`:
            (gpus: Sequence[int], main_args, task_option: TaskOption,
            register_func: Callable)
        """
        if run_task_func is None:
            run_task_func = run_task

        if self.is_distributed:
            import torch.multiprocessing as pt_mp
            from .ddp import launch_ddp_task

            n_gpus = len(self.args.use_gpus)
            pt_mp.spawn(
                fn=launch_ddp_task, nprocs=n_gpus, join=True,
                args=(n_gpus, run_task_func, self.args, self.task_option,
                      self.register_func, *run_task_func_args)
            )
        else:
            run_task_func(
                self.args.use_gpus, self.args, self.task_option,
                self.register_func,
            )
