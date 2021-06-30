from dataclasses import dataclass

from utils.log import info
from .base import OptionBase


@dataclass()
class TrainRoutine(OptionBase):
    epochs: int
    init_lr: float = None
    new_routine: bool = True
    optimizer_reset: bool = False
    note: str = None

    def set_init_lr(self, optimizer):
        if not self.new_routine:
            return False
        self.new_routine = False
        if self.init_lr is None:
            return False
        info(f'set init-lr to {self.init_lr} for new routine')
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.init_lr
            param_group['init_lr'] = self.init_lr
        return True
