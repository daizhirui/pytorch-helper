# Copyright (c) Zhirui Dai
from dataclasses import dataclass

from pytorch_helper.settings.space import Spaces
from .base import OptionBase


@dataclass()
class LossOption(OptionBase):
    name: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        """ build loss function
        """
        return Spaces.build_loss_fn(self.name, **self.kwargs)
