from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces

__all__ = ['DataloaderOption']


@dataclass()
class DataloaderOption(OptionBase):
    name: str
    kwargs: dict

    def build(self):
        """ build a dataloader
        """
        return Spaces.build_dataloader(self.name, **self.kwargs)
