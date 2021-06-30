from dataclasses import dataclass

import torch

from .base import OptionBase


@dataclass()
class OptimizerOption(OptionBase):
    name: str
    kwargs: dict

    def build(self, net):
        builder = getattr(torch.optim, self.name)
        return builder(
            filter(lambda p: p.requires_grad, net.parameters()), **self.kwargs
        )
