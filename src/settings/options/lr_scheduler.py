from dataclasses import dataclass

import torch

from .base import OptionBase


@dataclass()
class LRSchedulerOption(OptionBase):
    enable: bool
    name: str
    kwargs: dict

    def build(self, optimizer):
        if not self.enable:
            return None
        builder = getattr(torch.optim.lr_scheduler, self.name)
        return builder(optimizer, **self.kwargs)
