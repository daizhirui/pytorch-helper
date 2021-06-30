import os
from abc import ABC

import numpy as np
import torch
from torch.distributed import barrier
from torch.utils.tensorboard import SummaryWriter

from utils import log
from utils.io import save_pth
from utils.meter import Meter
from .base import TaskBase

__all__ = ['TrainTask']


class TrainTask(TaskBase, ABC):
    STAGE_TRAIN = 'train'
    STAGE_VALID = 'valid'
    STAGE_TEST = 'test'

    def _post_init(self, state_dict):
        assert self.option.train, \
            'TrainTask and its descendents is for task_option with train=True'
        self.optimizer = self.option.optimizer.build(self.unwrapped_model)
        self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)
        self.current_train_routine = None

        if state_dict and self.option.resume:
            # load from state_dict
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            except Exception as e:
                log.warn(repr(e))

            if self.lr_scheduler and state_dict['lr_scheduler']:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

            # miscellaneous
            self.cur_stage = self.STAGE_TRAIN
            self.loss_min = state_dict["loss_min"]
            self.epoch = state_dict["epoch"] + 1
            self.in_stage_meter_keys = state_dict['in_stage_meter_keys']
            np.random.set_state(state_dict["np_random_state"])
            torch.set_rng_state(state_dict["pt_random_state"].cpu())
            log.info(f"Resume from epoch {state_dict['epoch']}")

        # logging
        path = os.path.join(self.option.output_path_tb, 'meter_current.pkl')
        if os.path.exists(path):
            self.meter = Meter.load(path)
        else:
            self.meter = Meter()
        self.in_stage_meter_keys = set()

        # some logging should only happens on rank0
        if self.is_rank0:
            # progress bar
            self.progress_bars = {
                'epoch': log.bar(
                    initial=self.epoch,
                    total=self.option.train_setting.epochs, ncols=log.bar_len,
                    position=0, desc='Epoch'
                ),
                'train': log.bar(ncols=log.bar_len, position=1, desc='Train'),
                'valid': log.bar(ncols=log.bar_len, position=2, desc='Valid'),
                'test' : log.bar(ncols=log.bar_len, position=3, desc=' Test')
            }
            if getattr(self.dataloader, 'cross_valid', False):
                self.progress_bars['fold'] = log.bar(
                    ncols=log.bar_len, position=4, desc='Folds'
                )
            log.info("Initialize tensorboard")
            self.tboard = SummaryWriter(log_dir=self.option.output_path_tb)

    def save_pth(self, name=None):
        state_dict = self.get_state_dict()
        if name:
            path = os.path.join(self.option.output_path_pt, f"{name}.pth")
        else:
            path = os.path.join(self.option.output_path_pt,
                                f"epoch_{self.epoch}.pth")
        save_pth(path, state_dict)
        log.info(f"Saving checkpoint: {path} at epoch {self.epoch}")

    def _save_best_model(self, valid_loss):
        if isinstance(valid_loss, dict):
            if self.loss_min is None:
                self.loss_min = dict()
            for k, v in valid_loss.items():
                v_min = self.loss_min.get(k, None)
                if v_min is None or v_min > v:
                    self.loss_min[k] = v
                    self.save_pth(f'best-{k.replace("/", "-")}')
        else:
            if self.loss_min is None or self.loss_min > valid_loss:
                self.loss_min = valid_loss
                self.save_pth('best')

    def _backup(self, immediate=False):
        freq = self.option.train_setting.save_model_freq
        if immediate or (self.epoch > 0 and self.epoch % freq == 0):
            self.save_pth()
            path = os.path.join(self.option.output_path_tb, 'meter_current.pkl')
            self.meter.save(path)
            self.backup()

    def backup(self):
        raise NotImplementedError

    def one_epoch(self, epoch):
        self.epoch = epoch
        self.dataloader.set_epoch(epoch)

        result = self._train()
        summary = self._summarize_logging_after_stage()
        if self.is_rank0:
            self._rank0_summarize_logging_after_stage(result, summary)
        if self.is_distributed:
            barrier()

        result = self._valid()
        valid_summary = self._summarize_logging_after_stage()
        if self.is_rank0:
            self._rank0_summarize_logging_after_stage(result, valid_summary)
        if self.is_distributed:
            barrier()

        if self.option.train_setting.valid_on_test > 0 and \
                epoch % self.option.train_setting.valid_on_test == 0:
            result = self._test()
            summary = self._summarize_logging_after_stage()
            if self.is_rank0:
                self._rank0_summarize_logging_after_stage(result, summary)
            if self.is_distributed:
                barrier()
        return valid_summary

    def after_epoch(self, valid_summary):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.is_rank0:
            # save models
            self._save_best_model(valid_summary)
            self._backup()
            self.progress_bars['epoch'].update()
            self._rank0_update_logging_after_epoch()

    def run(self):
        try:
            self._run()
        except Exception as e:
            raise e
        finally:
            if self.is_rank0:
                self._backup(immediate=True)

    def _run(self):
        for epoch in range(self.epoch, self.option.train_setting.epochs):
            # NOTE: different stage should not share the same set of keys
            self.meter.reset(self.in_stage_meter_keys)
            valid_summary = self.one_epoch(epoch)
            self.after_epoch(valid_summary)

    def train(self, batch):
        # should return a dict of result to use
        self.optimizer.zero_grad()
        result = self.model_forward_backward(batch, backward=True)
        self.optimizer.step()
        return result

    def _train(self):
        self.cur_stage = self.STAGE_TRAIN

        self.model.train()
        self._setup_before_stage()
        dataloader = self.dataloader.train_loader
        self._setup_logging_before_stage(len(dataloader))

        with torch.enable_grad():
            for batch in dataloader:
                batch = self.load_batch(batch)
                result = self.train(batch)
                self._update_logging_in_stage(result)
        return result

    def valid(self, batch):
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch)

    def _valid(self):
        self.cur_stage = self.STAGE_VALID

        self.model.eval()
        self._setup_before_stage()
        dataloader = self.dataloader.valid_loader
        self._setup_logging_before_stage(len(dataloader))

        with torch.no_grad():
            for batch in dataloader:
                batch = self.load_batch(batch)
                result = self.valid(batch)
                self._update_logging_in_stage(result)
        return result

    def test(self, batch):
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch)

    def _test(self):
        self.cur_stage = self.STAGE_TEST

        self.model.eval()
        self._setup_before_stage()
        dataloader = self.dataloader.test_loader
        self._setup_logging_before_stage(len(dataloader))

        with torch.no_grad():
            for batch in dataloader:
                batch = self.load_batch(batch)
                result = self.test(batch)
                self._update_logging_in_stage(result)
        return result

    def load_batch(self, batch):
        # should return a dict of result to use
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=self.is_parallel)
        return batch

    def get_state_dict(self):
        lr = self.optimizer.param_groups[0]['lr']
        state_dict = dict(
            option=self.option,  # in case lose the config file
            in_stage_meter_keys=self.in_stage_meter_keys,
            model=self.unwrapped_model.state_dict(),
            loss_fn=self.loss_fn.state_dict(),
            optimizer=self.optimizer.state_dict(),
            lr=lr,
            lr_scheduler=None,
            epoch=self.epoch,
            loss_min=self.loss_min,
            rng_state=self.get_rng_state(),
        )
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def _setup_before_stage(self):
        raise NotImplementedError

    def _rank0_setup_logging_before_stage(self, stage_len):
        raise NotImplementedError

    def _setup_logging_before_stage(self, stage_len):
        if self.is_rank0:
            self.progress_bars[self.cur_stage].reset(stage_len)
            self._rank0_setup_logging_before_stage(stage_len)

    def _rank0_update_logging_in_stage(self, result):
        raise NotImplementedError

    def _update_logging_in_stage(self, result):
        loss = result['loss']
        if isinstance(loss, dict):
            for k, v in loss.items():
                if v is not None:
                    key = f'{self.cur_stage}/{k}'
                    self.meter.record(key, v.item())
                    self.in_stage_meter_keys.add(key)
        else:
            key = f'{self.cur_stage}/loss'
            self.meter.record_running_mean(key, loss.item(), result['batch_size'])
            self.in_stage_meter_keys.add(key)
        if self.is_rank0:
            self.progress_bars[self.cur_stage].update()
            self.progress_bars[self.cur_stage].refresh()
            self._rank0_update_logging_in_stage(result)

    def _summarize_logging_after_stage(self):
        summary = self.meter.mean([
            key for key in self.in_stage_meter_keys
            if key.startswith(self.cur_stage)
        ])
        for k, v in summary.items():
            tag = f'epoch-{k}'
            self.meter.record(tag, v)
            if self.is_rank0:
                self.tboard.add_scalar(tag, v, self.epoch)
                log.info(f'{tag} = {v}')
        return summary

    def _rank0_summarize_logging_after_stage(self, result, summary):
        raise NotImplementedError

    def _rank0_update_logging_after_epoch(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.tboard.add_scalar('learning-rate', lr, self.epoch)
