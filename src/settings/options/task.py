import os
import shutil
from dataclasses import InitVar
from dataclasses import dataclass
from typing import Union

from utils import log
from .base import OptionBase
from .dataloader import DataloaderOption
from .loss import LossOption
from .lr_scheduler import LRSchedulerOption
from .model import ModelOption
from .optimizer import OptimizerOption
from .train_setting import TrainSettingOption


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
    for_train: InitVar[bool]
    distributed: InitVar[bool]
    resume: bool = False

    def __post_init__(self, for_train, distributed):
        self.train = for_train
        # self.use_ddp = distributed
        if 'DATASET_PATH' in os.environ and 'OUTPUT_PATH' in os.environ:
            log.info("Setup dataset path and output path from environment"
                     "variables")
            self.dataset_path = os.path.abspath(os.environ['DATASET_PATH'])
            if for_train:
                self.output_path = os.path.abspath(os.environ['OUTPUT_PATH'])
            else:
                assert os.path.exists(self.output_path), \
                    f"{self.output_path} doesn't exist!"
        else:
            assert self.dataset_path is not None, \
                'dataset path is unavailable in environment or option file'
            assert self.output_path is not None, \
                'output path is unavailable in environment or option file'
            log.info("Setup dataset path and output path from option file")

        self.dataloader['kwargs']['root'] = self.dataset_path
        self.dataloader['kwargs']['use_ddp'] = distributed
        self.dataloader = DataloaderOption.load_from_dict(self.dataloader)
        self.loss = LossOption.load_from_dict(self.loss)
        self.optimizer = OptimizerOption.load_from_dict(self.optimizer)
        self.lr_scheduler = LRSchedulerOption.load_from_dict(self.lr_scheduler)
        self.model = ModelOption.load_from_dict(self.model)
        if type(self) is TaskOption:
            # no descendent class to initialize train_setting, use the default
            self.train_setting = TrainSettingOption.load_from_dict(
                self.train_setting
            )

        if self.datetime is None:
            self.datetime = log.get_datetime()
            while os.path.exists(self.output_path_tb):
                self.datetime = log.get_datetime()

        if for_train:
            log.info(f'create {self.output_path_tb}')
            os.makedirs(self.output_path_tb, exist_ok=True)
            log.info(f'create {self.output_path_pt}')
            os.makedirs(self.output_path_pt, exist_ok=True)
            self.save(os.path.join(self.output_path_tb, "..", "option.yaml"))

            if self.src_folder:
                dst = os.path.join(
                    self.output_path_pt, os.path.basename(self.src_folder)
                )
                if not os.path.exists(dst):
                    shutil.copytree(
                        self.src_folder, dst,
                        ignore=lambda _, x: [f for f in x if f == '__pycache__']
                    )
            else:
                log.warn(
                    f"src_folder is None. Strongly recommend you to specify the"
                    f" source code folder for automatic backup."
                )

    @property
    def output_path_pt(self):
        return os.path.join(self.output_path, self.name, self.datetime, "pt")

    @property
    def output_path_tb(self):
        return os.path.join(self.output_path, self.name, self.datetime, "tb")
