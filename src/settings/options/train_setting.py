from dataclasses import dataclass
from typing import List
from typing import Union

from .base import OptionBase
from .train_routine import TrainRoutine


@dataclass()
class TrainSettingOption(OptionBase):
    start_epoch: int
    epochs: int
    save_model_freq: int
    valid_on_test: bool
    train_routines: List[Union[dict, TrainRoutine]]
    gradient_clip: float = None

    def __post_init__(self):
        self.train_routines = [
            TrainRoutine(**r) for r in self.train_routines
        ]
        self.train_routines.sort(key=lambda x: x.epochs)

    def get_train_routine(self, epoch):
        last_epoch = 0
        for r in self.train_routines:
            if epoch < r.epochs:
                r.new_routine = epoch == last_epoch
                return r
            else:
                last_epoch = r.epochs
        raise ValueError(f'train routine for epoch {epoch} is missing')
