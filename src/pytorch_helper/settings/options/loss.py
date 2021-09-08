from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces

__all__ = ['LossOption']


@dataclass()
class LossOption(OptionBase):
    name: str
    kwargs: dict

    def __post_init__(self):
        super(LossOption, self).__post_init__()
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        """ build loss function
        """
        return Spaces.build_loss_fn(self.name, **self.kwargs)
