from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces


@dataclass()
class DataloaderOption(OptionBase):
    name: str
    kwargs: dict

    def build(self):
        return Spaces.build_dataloader(self.name, **self.kwargs)
