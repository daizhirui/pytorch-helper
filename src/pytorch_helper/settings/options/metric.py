from dataclasses import dataclass

from .base import OptionBase
from ..space import Spaces

__all__ = ['MetricOption']


@dataclass()
class MetricOption(OptionBase):
    name: str
    kwargs: dict

    def build(self):
        """ build a metric for test
        """
        return Spaces.build_metric(self.name, **self.kwargs)
