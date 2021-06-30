from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces


@dataclass()
class LossOption(OptionBase):
    name: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        return Spaces.build_loss_fn(self.name, **self.kwargs)
