# Copyright (c) Zhirui Dai
import os
from dataclasses import InitVar
from dataclasses import dataclass
from typing import Any
from typing import Type
from typing import TypeVar
from typing import Union

from pytorch_helper.utils import log
from pytorch_helper.utils.io import make_dirs
from pytorch_helper.utils.io import make_tar_file
from .base import OptionBase
from .dataloader import DataloaderOption
from .loss import LossOption
from .lr_scheduler import LRSchedulerOption
from .model import ModelOption
from .optimizer import OptimizerOption
from .train_setting import TrainSettingOption

T = TypeVar('T')


@dataclass()
class TaskOption(OptionBase):
    name: str
    type: str
    datetime: str
    notes: str
    output_path: str
    dataset_path: str
    train_setting: Union[dict, TrainSettingOption]
    dataloader: Union[dict, DataloaderOption]
    model: Union[dict, ModelOption]
    loss: Union[dict, LossOption]
    optimizer: Union[dict, OptimizerOption]
    lr_scheduler: Union[dict, LRSchedulerOption]
    src_folder: str
    resume: bool
    test_option: Any
    for_train: InitVar[bool]
    distributed: InitVar[bool]

    def __post_init__(self, for_train: bool, distributed: bool):
        self.gpu_ids = None
        self.train = for_train

        if 'DATASET_PATH' in os.environ and 'OUTPUT_PATH' in os.environ:
            log.info(
                __name__,
                'Setup dataset path and output path from environment variables'
            )
            self.dataset_path = os.path.abspath(os.environ['DATASET_PATH'])
            self.output_path = os.path.abspath(os.environ['OUTPUT_PATH'])
        else:
            assert self.dataset_path is not None, \
                'dataset path is unavailable in environment or option file'
            assert self.output_path is not None, \
                'output path is unavailable in environment or option file'

            log.info(
                __name__,
                'Setup dataset path and output path from option file'
            )

            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(
                    f'dataset path: {self.dataset_path} does not exist'
                )
            if not os.path.isdir(self.output_path):
                raise NotADirectoryError(
                    f'output path: {self.output_path} does not exist or '
                    f'is not a folder'
                )

        if isinstance(self.dataloader, dict):
            self.dataloader['kwargs']['root'] = self.dataset_path
            self.dataloader['kwargs']['use_ddp'] = distributed
            self.dataloader = DataloaderOption.load_from_dict(self.dataloader)

        self.loss = self.load_option(self.loss, LossOption)
        self.optimizer = self.load_option(self.optimizer, OptimizerOption)
        self.lr_scheduler = self.load_option(
            self.lr_scheduler, LRSchedulerOption
        )
        self.model = self.load_option(self.model, ModelOption)
        self.train_setting = self.load_option(
            self.train_setting, TrainSettingOption
        )

        if self.datetime is None:
            self.datetime = log.get_datetime()
            while os.path.exists(self.output_path_tb):
                self.datetime = log.get_datetime()

        if for_train:
            log.info(__name__, f'create {self.output_path_tb}')
            make_dirs(self.output_path_tb)
            log.info(__name__, f'create {self.output_path_pth}')
            make_dirs(self.output_path_pth)
            self.save_as_yaml(os.path.join(self.output_path_tb, '..', 'option.yaml'))

            if self.src_folder:
                dst = os.path.join(
                    self.output_path_pth,
                    os.path.basename(self.src_folder) + '.tar.gz'
                )
                make_tar_file(self.src_folder, dst)
            else:
                log.warn(
                    __name__,
                    f'src_folder is None. Strongly recommend you to specify the'
                    f' source code folder for automatic backup.'
                )

    @staticmethod
    def load_option(option_dict: dict, option_cls: Type[T]) -> T:
        """ convert option_dict to option_cls if option_dict is a dict

        :param option_dict: dict to convert
        :param option_cls: class of option to convert to
        :return: an instance of option_cls
        """
        if isinstance(option_dict, dict):
            return option_cls.load_from_dict(option_dict)
        else:
            return option_dict

    @property
    def output_path_pth(self) -> str:
        """
        :return: the path of the checkpoint folder
        """
        return os.path.join(self.output_path, self.name, self.datetime, 'pth')

    @property
    def output_path_tb(self) -> str:
        """
        :return: the path of tensorboard folder
        """
        return os.path.join(self.output_path, self.name, self.datetime, 'tb')