# Copyright (c) Zhirui Dai
import os
from abc import ABC

import numpy as np
import torch

from pytorch_helper.settings.options.train_routine import TrainRoutine
from pytorch_helper.utils import log
from pytorch_helper.utils.dist import synchronize
from pytorch_helper.utils.io import make_dirs
from pytorch_helper.utils.io import save_dict_as_csv
from .train import TrainTask

__all__ = ['TestTask']


class TestTask(TrainTask, ABC):

    def __init__(self, task_option):
        self.output_path_test = None
        super(TestTask, self).__init__(task_option)

        self.keep_model_output = True
        self.cur_stage = self.STAGE_TEST
        self.model_output_dict = dict()
        self.datetime_test = log.get_datetime()

    def post_init(self, state_dict):
        if self.is_rank0:
            # progress bar
            self.progress_bars = {
                self.STAGE_TEST: log.pbar(position=0, desc=' Test')
            }

        epoch = 'NA'
        if state_dict:
            epoch = state_dict.get('epoch', 'NA')
        self.output_path_test = os.path.join(
            self.option.output_path_pth, f'test-epoch-{epoch}'
        )
        make_dirs(self.output_path_test)

    def update_logging_in_stage(self, result):
        if self.keep_model_output:
            model_output = result['pred']
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
                log.warn(
                    __name__,
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
            path = os.path.join(
                self.output_path_test, f'test-summary.{self.datetime_test}.csv'
            )
            save_dict_as_csv(summary, path)
        synchronize()

        for key, value in self.model_output_dict.items():
            self.model_output_dict[key] = np.concatenate(value, axis=0)

        return summary
