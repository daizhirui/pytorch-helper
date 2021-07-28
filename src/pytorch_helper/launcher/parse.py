import os
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass
from typing import Type
from typing import TypeVar

from ..utils.log import info

T = TypeVar('T')

__all__ = ['MainArg']


@dataclass()
class MainArg:
    task_option_file: str
    test_option_file: str
    use_gpus: list
    wait_gpus: bool
    pth_path: str
    resume: bool
    dataset_path: str
    output_path: str
    ddp_port: int
    use_data_parallel: bool
    boost: bool
    debug: bool
    debug_size: int

    def __post_init__(self):
        os.environ['DDP_PORT'] = str(self.ddp_port)
        self.use_gpus.sort()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.use_gpus))
        self.cuda_device_mapping = OrderedDict(enumerate(self.use_gpus))
        self.use_gpus = list(range(len(self.use_gpus)))

        os.environ['DEBUG'] = '1' if self.debug else '0'
        os.environ['DEBUG_SIZE'] = str(self.debug_size)

        import torch
        if self.boost:
            info(__name__, 'turn on cudnn boost')
            cudnn = getattr(torch.backends, 'cudnn', None)
            if cudnn:
                cudnn.deterministic = False
                cudnn.benchmark = True

    @staticmethod
    def get_parser() -> ArgumentParser:
        """ Construct a ArgumentParser

        :return: ArgumentParser to parse arguments
        """
        parser = ArgumentParser('PyTorch-Helper Argument Parser')
        group = parser.add_argument_group(
            'Option File arguments', 'Specify path to option files'
        )
        group.add_argument(
            '--task-option-file',
            type=str,
            help='Path to the file of training options')
        group.add_argument(
            '--test-option-file',
            type=str,
            help='Path to the file of extra options for testing')

        group = parser.add_argument_group(
            'GPU Setting arguments', 'Specify GPU settings'
        )
        group.add_argument(
            '--use-gpus',
            nargs='+', required=True, type=int, metavar='GPU_INDEX',
            help='Indices of GPUs for training')
        group.add_argument(
            '--wait-gpus',
            action='store_true',
            help='Wait for selected GPUs to be ready before training')
        group.add_argument(
            '--boost',
            action='store_true',
            help='Turn on cudnn boost')
        group = group.add_mutually_exclusive_group()
        group.add_argument(
            '--ddp-port',
            default=23456, type=int,
            help='Port used for DistributedDataParallel, default: 23456')
        group.add_argument(
            '--use-data-parallel',
            action='store_true',
            help='Use DataParallel instead of DistributedDataParallel')

        group = parser.add_argument_group(
            'Resume Setting arguments', 'Specify resume settings'
        )
        group.add_argument(
            '--pth-path', '--pth-file',
            type=str,
            help="Path to the pt file to resume training")
        group.add_argument(
            '--resume', action='store_true',
            help='Resume training from the pth-file given by --pth-path'
        )

        group = parser.add_argument_group(
            'Path Setting arguments', 'Specify path settings'
        )
        group.add_argument(
            '--dataset-path',
            type=str,
            help='Path to the dataset')
        group.add_argument(
            '--output-path',
            type=str,
            help='Path to save the training')

        parser.add_argument(
            '--debug',
            action='store_true',
            help='If used, set environment variable DEBUG=1')
        parser.add_argument(
            '--debug-size', default=32,
            help='Number of samples in each stage dataset for debug'
        )
        return parser

    @classmethod
    def parse(cls: Type[T]) -> T:
        """ Parse arguments from command line, build `MainArg` or its
        descendent, and do some setups such as setting `os.environ`.

        :return: MainArg or its descendent
        """
        parser = cls.get_parser()
        args = parser.parse_args()

        args = cls(**vars(args))

        return args
