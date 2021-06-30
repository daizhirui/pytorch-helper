import csv
import os
from abc import ABC

import h5py
import numpy as np
from torch.distributed import barrier

from settings.options.train_routine import TrainRoutine
from utils import log
from utils.log import get_datetime
from utils.meter import Meter
from .train import TrainTask

__all__ = ['TestTask']


class TestTask(TrainTask, ABC):

    def _post_init(self, state_dict):
        self.epoch = state_dict['epoch']
        # logging
        src = os.path.join(self.option.output_path_tb, 'meter_current.pkl')
        if os.path.exists(src):
            self.meter = Meter.load(src)
        else:
            self.meter = Meter()
        self.in_stage_meter_keys = set()

        # logging should only happen on rank0 when `DistributedDataParallel` is used
        if self.is_rank0:
            # progress bar
            self.progress_bars = {
                'test': log.bar(ncols=log.bar_len, position=0, desc=' Test')
            }

        self.keep_model_output = True
        self.output_path_test = os.path.join(
            self.option.output_path, self.option.name, self.option.datetime, 'test'
        )
        os.makedirs(self.output_path_test, exist_ok=True)
        self.model_output_file = os.path.join(
            self.output_path_test, f'model_output_{self.rank}.h5'
        )
        self.model_output_dict = dict()

    def _update_logging_in_stage(self, result):
        super(TestTask, self)._update_logging_in_stage(result)
        if self.keep_model_output:
            result = result['pred']
            for key, data in result.items():
                if data is not None:
                    self.model_output_dict.setdefault(key, list()).append(
                        data.cpu().numpy()
                    )

    def save_model_output(self):
        if self.keep_model_output:
            file = h5py.File(self.model_output_file, 'w')
            for name, data in self.model_output_dict.items():
                data = np.concatenate(data, axis=0)
                file.create_dataset(
                    name=name, shape=data.shape, dtype=data.dtype, data=data,
                    compression='gzip'
                )
            file.attrs["task_name"] = self.option.name
            file.attrs["datetime"] = self.option.datetime
            file.attrs["pt_path"] = self.option.model.pt_path
            file.close()
        else:
            log.info(f"model output is not kept!")

    def save_summary(self, summary, file_path=None, append=False):
        if not self.is_rank0:
            return

        if file_path is None:
            file_path = os.path.join(
                self.output_path_test, f"summary-{get_datetime()}.csv"
            )

        new_csv = True
        if os.path.exists(file_path) and append:
            new_csv = False
            with open(file_path, 'r') as file:
                fieldnames = file.readline().strip().split(',')
        else:
            fieldnames = ['name', 'datetime', 'epoch', 'pt_file'] + \
                         sorted(list(summary.keys()))

        with open(file_path, 'a+' if append else 'w', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
            if new_csv:
                csv_writer.writeheader()
            row = dict(
                name=self.option.name,
                datetime=self.option.datetime,
                epoch=self.epoch,
                pt_file=os.path.basename(self.option.model.pt_path),
                **summary
            )
            for k in fieldnames:
                if k not in row:
                    row[k] = None
            csv_writer.writerow(row)

    def _summarize_logging_after_stage(self):
        summary = self.meter.mean([
            key for key in self.in_stage_meter_keys
            if key.startswith(self.cur_stage)
        ])
        for k, v in summary.items():
            tag = f'epoch-{k}'
            self.meter.record(tag, v)
        return summary

    def run(self):
        self.current_train_routine = TrainRoutine(epochs=self.epoch)
        result = self._test()
        summary = self._summarize_logging_after_stage()
        if self.is_rank0:
            self._rank0_summarize_logging_after_stage(result, summary)
        if self.is_distributed:
            barrier()
        return summary
