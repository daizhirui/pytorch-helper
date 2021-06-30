import os
from dataclasses import dataclass

from utils.io import load_pth
from utils.log import info
from .base import OptionBase
from ..space import Spaces


@dataclass()
class ModelOption(OptionBase):
    name: str
    kwargs: dict
    pth_path: str

    @property
    def pth_available(self):
        if self.pth_path and len(self.pth_path) > 0:
            return os.path.isfile(self.pth_path)
        return False

    def build(self):
        model = Spaces.build_model(self.name, **self.kwargs)
        info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = load_pth(self.pth_path)
            model.load_state_dict(state_dict['model'])
            info(f'Load model state from {self.pth_path}')
        return model, state_dict
