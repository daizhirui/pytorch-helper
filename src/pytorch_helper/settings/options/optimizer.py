# Copyright (c) Zhirui Dai
from dataclasses import dataclass

import torch

from .base import OptionBase


@dataclass()
class OptimizerOption(OptionBase):
    name: str
    kwargs: dict

    def build(self, model):
        """ build the optimizer with the given model

        :param model: the model used to build the optimizer, only parameters
            whose `requires_grad` is True will be posted to the optimizer
        :return: the optimizer
        """
        builder = getattr(torch.optim, self.name)
        return builder(
            filter(lambda p: p.requires_grad, model.parameters()), **self.kwargs
        )
