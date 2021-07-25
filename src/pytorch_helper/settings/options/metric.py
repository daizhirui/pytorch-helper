# Copyright (c) Zhirui Dai
from dataclasses import dataclass

from pytorch_helper.settings.space import Spaces
from .base import OptionBase


@dataclass()
class MetricOption(OptionBase):
    name: str
    kwargs: dict

    def build(self):
        """ build a metric for test
        """
        return Spaces.build_metric(self.name, **self.kwargs)
