# Copyright (c) Zhirui Dai
import os
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from torch.nn.modules import Module

from pytorch_helper.settings.space import Spaces
from pytorch_helper.utils.io import load_pth
from pytorch_helper.utils.log import info
from .base import OptionBase


@dataclass()
class ModelOption(OptionBase):
    name: str
    kwargs: dict
    pth_path: str

    @property
    def pth_available(self):
        """ check if `self.pth_path` is available

        :return: Bool to indicate if the checkpoint file is available
        """
        if self.pth_path:
            if os.path.isfile(self.pth_path):
                return True
            elif os.path.isdir(self.pth_path):
                raise IsADirectoryError(
                    f'pth path "{self.pth_path}" should not be a folder'
                )
            else:
                raise FileNotFoundError(
                    f'{self.pth_path} does not exist'
                )
        return False

    def build(self) -> Tuple[Module, Optional[dict]]:
        """ build the model and load the weights from the checkpoint if
        `self.pth_available` is True.

        :return: model and state_dict
        """
        model = Spaces.build_model(self.name, **self.kwargs)
        info(__name__, f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = load_pth(self.pth_path)
            model.load_state_dict(state_dict['model'])
            info(__name__, f'Load model state from {self.pth_path}')
        return model, state_dict
