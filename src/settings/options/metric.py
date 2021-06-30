from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces


@dataclass()
class MetricOption(OptionBase):
    name: str
    build_kwargs: dict

    def build(self):
        return Spaces.build_metric(self.name, **self.build_kwargs)
