import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List
from typing import TypeVar

from utils.log import info

T = TypeVar('T')


@dataclass()
class MainArg:
    option_file: str
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
    default = None

    @classmethod
    def parse(cls: T) -> T:
        parser = ArgumentParser()
        parser.add_argument(
            "--option-file",
            default="option.yaml", type=str,
            help="Path to the file of training options")
        parser.add_argument(
            "--use-gpus",
            nargs='+', required=True, type=List[int],
            help="Indices of GPUs for training")
        parser.add_argument(
            '--wait-gpus',
            action='store_true',
            help='Wait for selected GPUs to be ready before training')
        parser.add_argument(
            "--pth-path",
            type=str,
            help="Path to the pt file to resume training")
        parser.add_argument(
            "--resume", action='store_true',
            help="Resume training from the pt-file given by --pth-path"
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            help="Path to the dataset")
        parser.add_argument(
            "--output-path",
            type=str,
            help="Path to save the training")
        parser.add_argument(
            '--ddp-port',
            default=23456, type=int,
            help='Port used for DistributedDataParallel')
        parser.add_argument(
            '--use-data-parallel',
            action='store_true',
            help='Use DataParallel instead of DistributedDataParallel')
        parser.add_argument(
            '--boost',
            action='store_true',
            help='turn on cudnn boost')
        parser.add_argument(
            '--debug',
            action='store_true',
            help='if used, set environment variable DEBUG=1')

        arg = parser.parse_args(sys.argv[1:])
        arg.use_gpus = [x.lower() for x in arg.use_gpus]
        arg.use_gpus.sort(key=lambda x: int(x))

        if 'cuda' in arg.use_gpus:
            arg.use_gpus = [0]
            info('Use gpu 0 only because "cuda" is specified')
        elif 'cpu' in arg.use_gpus:
            arg.use_gpus = ['cpu']
            info('Use cpu only because "cpu" is specified')
        else:
            if len(arg.use_gpus) > 1 and not arg.use_data_parallel:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(arg.use_gpus)
                os.environ['DDP_PORT'] = str(arg.ddp_port)
            arg.use_gpus = sorted([int(x) for x in arg.use_gpus])

        os.environ['DEBUG'] = '1' if arg.debug else '0'

        args = cls(**arg.__dict__)
        MainArg.default = args
        return args
