import os
from abc import ABC
from collections import OrderedDict

import numpy as np
import torch

from .base import BatchPack
from .train import TrainTask
from ..settings.options.train_routine import TrainRoutine
from ..utils.dist import synchronize
from ..utils.io import make_dirs
from ..utils.io import save_dict_as_csv
from ..utils.log import get_datetime
from ..utils.log import get_logger
from ..utils.log import pbar

__all__ = ['TestTask']

logger = get_logger(__name__)


class TestTask(TrainTask, ABC):

    def __init__(self, task_option):
        self.output_path_test = None
        super(TestTask, self).__init__(task_option)

        self.keep_model_output = True
        self.cur_stage = self.STAGE_TEST
        self.model_output_dict = dict()
        self.datetime_test = get_datetime()

    def post_init(self, state_dict):
        if self.is_rank0:
            # progress bar
            self.progress_bars = {
                self.STAGE_TEST: pbar(position=0, desc=' Test')
            }

        self.epoch = -1
        if state_dict is not None:
            self.epoch = state_dict.get('epoch', -1)

        self.output_path_test = os.path.realpath(os.path.join(
            self.option.output_path_pth, '..', 'test'
        ))
        make_dirs(self.output_path_test)

    def update_logging_in_stage(self, result: BatchPack):
        if self.keep_model_output:
            model_output = result.pred
            if isinstance(model_output, dict):
                for key, data in model_output.items():
                    if data is not None:
                        self.model_output_dict.setdefault(key, list()).append(
                            data.cpu().numpy()
                        )
            elif isinstance(model_output, torch.Tensor):
                self.model_output_dict.setdefault('output', list()).append(
                    model_output.cpu().numpy()
                )
            else:
                logger.warn(

                    f'unable to save model output of type {type(model_output)}'
                )
        super(TestTask, self).update_logging_in_stage(result)

    def setup_before_stage(self):
        self.cur_stage = self.STAGE_TEST
        self.current_train_routine = TrainRoutine(epochs=self.epoch)
        super(TestTask, self).setup_before_stage()

    def run(self):
        self._test()
        summary = self.summarize_logging_after_stage()
        if self.is_rank0:
            path = os.path.join(self.output_path_test, 'test-summary.csv')
            save_dict_as_csv(path, summary)
        synchronize()

        for key, value in self.model_output_dict.items():
            self.model_output_dict[key] = np.concatenate(value, axis=0)

        return summary

    def summarize_logging_after_stage(self) -> OrderedDict:
        summary = OrderedDict()
        summary['name'] = self.option.name
        summary['datetime'] = self.option.datetime
        summary['epoch'] = self.epoch
        if self.option.model.pth_path is None:
            summary['pth_file'] = 'None'
        else:
            summary['pth_file'] = os.path.basename(
                self.option.model.pth_path)
        summary.update(super(TestTask, self).summarize_logging_after_stage())
        return summary
