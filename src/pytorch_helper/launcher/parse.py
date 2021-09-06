import os
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass
from typing import Type
from typing import TypeVar
from typing import Union

from ..utils.gpu.wait_gpus import set_cuda_visible_devices
from ..utils.log import get_logger

T = TypeVar('T')

__all__ = ['MainArg']

logger = get_logger(__name__)


@dataclass()
class MainArg:
    task_option_file: str
    test_option_file: str
    use_gpus: list
    wait_gpus: Union[bool, list]
    pth_path: str
    resume: bool
    dataset_path: str
    output_path: str
    ddp_port: int
    use_data_parallel: bool
    boost: bool
    img_ext: str
    debug: bool
    debug_size: int

    def __post_init__(self):
        os.environ['DDP_PORT'] = str(self.ddp_port)
        if len(self.wait_gpus) > 0:
            assert len(self.wait_gpus) == len(self.use_gpus), \
                '--wait-gpus should post the same number of GPUs as --use-gpus'
            nvidia_smi_mapping = OrderedDict(zip(self.use_gpus, self.wait_gpus))
            self.use_gpus.sort()
            self.cuda_device_mapping = OrderedDict(
                (i, nvidia_smi_mapping[x]) for i, x in enumerate(self.use_gpus)
            )
            self.wait_gpus = True
        else:
            self.use_gpus.sort()
            self.cuda_device_mapping = None
            self.wait_gpus = False
        set_cuda_visible_devices(self.use_gpus)
        self.use_gpus = list(range(len(self.use_gpus)))

        os.environ['DEBUG'] = '1' if self.debug else '0'
        os.environ['DEBUG_SIZE'] = str(self.debug_size)

        import torch
        if self.boost:
            logger.info('turn on cudnn boost')
            cudnn = getattr(torch.backends, 'cudnn', None)
            if cudnn:
                cudnn.deterministic = False
                cudnn.benchmark = True

        from pytorch_helper.utils import io
        io.config['img_ext'] = self.img_ext

    @staticmethod
    def get_parser() -> ArgumentParser:
        """ Construct a ArgumentParser

        :return: ArgumentParser to parse arguments
        """
        parser = ArgumentParser()
        group = parser.add_argument_group(
            'Option File arguments', 'Specify path to option files'
        )
        group.add_argument(
            '--task-option-file', type=str,
            help='Path to the file of training options')
        group.add_argument(
            '--test-option-file', type=str,
            help='Path to the file of extra options for testing')

        group = parser.add_argument_group(
            'GPU Setting arguments', 'Specify GPU settings'
        )
        group.add_argument(
            '--use-gpus',
            nargs='+', required=True, type=int, metavar='GPU_INDEX',
            help='Indices of GPUs for training')
        group.add_argument(
            '--wait-gpus', nargs='+', default=[], type=int, metavar='GPU_INDEX',
            help='NVIDIA-SMI Indices of GPUs specified by --use-gpus to wait. '
                 'Note that these indices may be different from the ones '
                 'posted to --use-gpus if NVIDIA-SMI presents an abnormal '
                 'device mapping. e.g. GPU4 shown in NVIDIA-SMI might be '
                 'GPU8 in --use-gpus. This is abnormal but --wait-gpus can '
                 'deal with it correctly if you post the correct mapping by '
                 '"--use-gpus 8 --wait-gpus 4".')
        group.add_argument(
            '--boost', action='store_true',
            help='Turn on cudnn boost')
        group = group.add_mutually_exclusive_group()
        group.add_argument(
            '--ddp-port', default=23456, type=int,
            help='Port used for DistributedDataParallel, default: 23456')
        group.add_argument(
            '--use-data-parallel',
            action='store_true',
            help='Use DataParallel instead of DistributedDataParallel')

        group = parser.add_argument_group(
            'Resume Setting arguments', 'Specify resume settings'
        )
        group.add_argument(
            '--pth-path', '--pth-file', type=str,
            help="Path to the pt file to resume training")
        group.add_argument(
            '--resume', action='store_true',
            help='Resume training from the pth-file given by --pth-path'
        )

        group = parser.add_argument_group(
            'IO Setting arguments', 'Specify input output settings'
        )
        group.add_argument(
            '--dataset-path', type=str,
            help='Path to the dataset'
        )
        group.add_argument(
            '--output-path', type=str,
            help='Path to save the training'
        )
        group.add_argument(
            '--img-ext', type=str, default='png',
            help='File extension of saved images'
        )

        group = parser.add_argument_group(
            'Debug Setting arguments', 'Specify debug settings'
        )
        group.add_argument(
            '--debug',
            action='store_true',
            help='If used, set environment variable DEBUG=1')
        group.add_argument(
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
