# Copyright (c) Zhirui Dai
from dataclasses import dataclass

from pytorch_helper.settings.space import Spaces
from .base import OptionBase


@dataclass()
class DataloaderOption(OptionBase):
    name: str
    kwargs: dict

    def build(self):
        """ build a dataloader
        """
        return Spaces.build_dataloader(self.name, **self.kwargs)
