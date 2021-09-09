import os
from dataclasses import InitVar
from dataclasses import dataclass
from typing import Any
from typing import Type
from typing import TypeVar
from typing import Union

from .base import OptionBase
from .dataloader import DataloaderOption
from .loss import LossOption
from .lr_scheduler import LRSchedulerOption
from .model import ModelOption
from .optimizer import OptimizerOption
from .train_setting import TrainSettingOption
from ...utils.io import make_dirs
from ...utils.io import make_tar_file
from ...utils.log import get_datetime
from ...utils.log import get_logger

T = TypeVar('T')

__all__ = [
    'merge_task_dicts',
    'TaskOption'
]

logger = get_logger(__name__)


def merge_task_dicts(base: dict, update: dict):
    out = base.copy()

    for key, value in update.items():
        assert key in base, f'key {key} not found in base dict'
        if isinstance(value, dict):
            out[key] = merge_task_dicts(base[key], value)
        else:
            out[key] = value
    return out


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
    is_distributed: InitVar[bool]
    print_freq: int = 10

    def __post_init__(self, for_train: bool, is_distributed: bool):
        super(TaskOption, self).__post_init__()
        self.cuda_ids = None
        self.train = for_train

        if 'DATASET_PATH' in os.environ and 'OUTPUT_PATH' in os.environ:
            logger.info(
                'Setup dataset path and output path from environment variables'
            )
            self.dataset_path = os.path.abspath(os.environ['DATASET_PATH'])
            self.output_path = os.path.abspath(os.environ['OUTPUT_PATH'])
        else:
            assert self.dataset_path is not None, \
                'dataset path is unavailable in environment or option file'
            assert self.output_path is not None, \
                'output path is unavailable in environment or option file'

            logger.info(
                'Setup dataset path and output path from option file'
            )

            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(
                    f'dataset path: {self.dataset_path} does not exist'
                )
            if os.path.exists(self.output_path):
                if not os.path.isdir(self.output_path):
                    raise NotADirectoryError(
                        f'output path: {self.output_path} does not exist or '
                        f'is not a folder'
                    )
            else:
                make_dirs(self.output_path)

        if isinstance(self.dataloader, dict):
            self.dataloader['kwargs']['root'] = self.dataset_path
            self.dataloader['kwargs']['use_ddp'] = is_distributed
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
            self.datetime = get_datetime()
            while os.path.exists(self.output_path_tb):
                self.datetime = get_datetime()

        if for_train:
            logger.info(f'create {self.output_path_tb}')
            make_dirs(self.output_path_tb)
            logger.info(f'create {self.output_path_pth}')
            make_dirs(self.output_path_pth)
            self.save_as_yaml(
                os.path.join(self.output_path_tb, '..', 'option.yaml'))

            if self.src_folder:
                dst = os.path.join(
                    self.output_path_pth,
                    os.path.basename(self.src_folder) + '.tar.gz'
                )
                make_tar_file(self.src_folder, dst)
            else:
                logger.warn(
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
    def output_path_task(self):
        return os.path.join(self.output_path, self.name, self.datetime)

    @property
    def output_path_pth(self) -> str:
        """
        :return: the path of the checkpoint folder
        """
        return os.path.join(self.output_path_task, 'pth')

    @property
    def output_path_tb(self) -> str:
        """
        :return: the path of tensorboard folder
        """
        return os.path.join(self.output_path_task, 'tb')
