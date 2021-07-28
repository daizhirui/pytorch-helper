import os
from abc import ABC
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import BatchPack
from .base import TaskBase
from ..utils.dist import synchronize
from ..utils.io import save_pth
from ..utils.log import info
from ..utils.log import pbar
from ..utils.meter import Meter

__all__ = ['TrainTask']


class TrainTask(TaskBase, ABC):

    def __init__(self, task_option):
        self.progress_bars = None
        self.batch_cnt = {
            self.STAGE_TRAIN: 0,
            self.STAGE_VALID: 0,
            self.STAGE_TEST : 0,
            'all'           : 0
        }
        self.in_stage_meter_keys = set()

        super(TrainTask, self).__init__(task_option)

        self.optimizer = self.option.optimizer.build(self.unwrapped_model)
        self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)

        # logging
        path = os.path.join(self.option.output_path_tb, 'meter_current.pkl')
        if os.path.exists(path) and self.option.resume:
            self.meter = Meter.load(path)
        else:
            self.meter = Meter()
        self.in_stage_logged = False

    def post_init(self, state_dict: dict):
        # some logging should only happens on rank0
        if self.is_rank0:
            # progress bar
            self.progress_bars = {
                'epoch'         : pbar(
                    initial=self.epoch, total=self.option.train_setting.epochs,
                    position=0, desc='Epoch'
                ),
                self.STAGE_TRAIN: pbar(
                    position=1, desc='Train'
                ),
                self.STAGE_VALID: pbar(
                    position=2, desc='Valid'
                ),
                self.STAGE_TEST : pbar(
                    position=3, desc=' Test'
                )
            }

        if self.is_rank0:
            info(__name__, "Initialize tensorboard")
            self.tboard = SummaryWriter(log_dir=self.option.output_path_tb)

        if state_dict and self.option.resume:
            self.load_state(state_dict, 'optimizer', self.optimizer)
            self.load_state(state_dict, 'lr_scheduler', self.lr_scheduler)
            if 'rng' in state_dict:
                self.set_rng_state(state_dict['rng'])

            # miscellaneous
            self.loss_min = state_dict.get('loss_min', self.loss_min)
            self.batch_cnt = state_dict.get('batch_cnt', self.batch_cnt)
            self.in_stage_meter_keys = state_dict.get(
                'in_stage_meter_keys', self.in_stage_meter_keys
            )
            if 'epoch' in state_dict:
                self.epoch = state_dict['epoch'] + 1
                if self.is_rank0:
                    self.progress_bars['epoch'].update(self.epoch)
                info(__name__, f"Resume from epoch {state_dict['epoch']}")

    def state_dict(self):
        state_dict = super(TrainTask, self).state_dict()
        state_dict.update(dict(
            lr=self.optimizer.param_groups[0]['lr'],
            optimizer=self.get_state(self.optimizer),
            lr_scheduler=self.get_state(self.lr_scheduler),
            loss_min=self.loss_min,
            in_stage_meter_keys=self.in_stage_meter_keys,
            batch_cnt=self.batch_cnt
        ))
        return state_dict

    def save_pth(self, name: str = None, resumable: bool = True):
        """ save the task state as a checkpoint

        :param name: str of the checkpoint file name, default is `epoch_x` where
            x is the current epoch number
        :param resumable: Bool to determine whether to store extra states such
            that a training process can resume from the checkpoint file.
        """
        if resumable:
            state_dict = self.state_dict()
        else:
            state_dict = super(TrainTask, self).state_dict()

        pth_name = f'{name}.pth' if name else f'epoch_{self.epoch}.pth'
        path = os.path.join(self.option.output_path_pth, pth_name)
        save_pth(path, state_dict)
        info(__name__, f"Saving checkpoint: {path} at epoch {self.epoch}")

    def save_best_model(self, valid_loss):
        """ save the model which has minimum validation loss

        :param valid_loss: latest validation loss
        """
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

    def backup(self, immediate: bool = False, resumable: bool = True):
        """ determine if it is time to backup the task and save the task state
        if necessary

        :param immediate: Bool to ignore all the restriction, save the task
            state and the meter state
        :param resumable: Bool to set the checkpoint file resumable
        """
        freq = self.option.train_setting.save_model_freq
        if immediate or (self.epoch > 0 and self.epoch % freq == 0):
            self.save_pth(resumable=resumable)
            path = f'meter-{"train" if self.option.train else "test"}.pkl'
            path = os.path.join(self.option.output_path_tb, path)
            self.meter.save(path)

    def one_epoch(self, epoch: int) -> dict:
        """ run a epoch of the task, including training, validation and test if
        `self.option.train_setting.valid_on_test` is positive integer.

        :param epoch: int of current epoch number
        :return: dict of validation result
        """
        self.epoch = epoch
        self.dataloader.set_epoch(epoch)

        self._train()
        self.summarize_logging_after_stage()
        synchronize()

        self._valid()
        valid_summary = self.summarize_logging_after_stage()
        synchronize()

        if self.option.train_setting.valid_on_test > 0 and \
                epoch % self.option.train_setting.valid_on_test == 0:
            self._test()
            self.summarize_logging_after_stage()
            synchronize()

        return valid_summary

    def after_epoch(self, valid_summary: dict):
        """ do some setup and logging with the validation summary after one
        epoch, such as learning rate adjustment, save the best model, etc.

        :param valid_summary:
        :return:
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.is_rank0:
            # save models
            self.save_best_model(valid_summary)
            self.backup()
            lr = self.optimizer.param_groups[0]['lr']
            self.progress_bars['epoch'].update()
            if self.tboard is not None:
                self.tboard.add_scalar('learning-rate', lr, self.epoch)
            self.rank0_update_logging_after_epoch()

    def run(self):
        """ the entry to launch the task, run the task until `self.epoch` is
        equal to `self.option.training_setting.epochs`, and save the final
        model state as a non-resumable checkpoint file.
        """
        for epoch in range(self.epoch, self.option.train_setting.epochs):
            # NOTE: different stage should not share the same set of keys
            self.meter.reset_tags(self.in_stage_meter_keys)
            valid_summary = self.one_epoch(epoch)
            self.after_epoch(valid_summary)
        # save only the state dicts of model and loss_fn
        self.save_pth('model_final', resumable=False)

    def setup_before_stage(self):
        """ do some setup before a stage, training, validation or testing.
        `self.in_stage_logged` will be set to False. The model will be set for
        training or non-training properly. And `self.cur_dataloader` is also
        switched accordingly. If the stage is `STAGE_TRAIN`,
        `self.current_train_routine` may be updated and makes some changes to
        the model like freezing or unfreezing some modules.
        """
        self.in_stage_logged = False
        if self.cur_stage == self.STAGE_TRAIN:
            self.current_train_routine = \
                self.option.train_setting.get_train_routine(self.epoch)
            self.model.train()
            self.cur_dataloader = self.dataloader.train_loader
            if self.current_train_routine.set_init_lr(self.optimizer):
                if self.is_rank0 and self.epoch > 0:
                    info(__name__, "Save before applying new routine")
                    self.epoch -= 1
                    self.save_pth(f'epoch_{self.epoch}')
                    self.epoch += 1
            if self.current_train_routine.train_modules is not None:
                self.freeze_and_unfreeze_modules(
                    self.current_train_routine.train_modules,
                    # optimizers like Adam still change the frozen weight
                    # because they are using statistics of gradients
                    reset_optimizer=self.current_train_routine.optimizer_reset
                )
                self.current_train_routine.optimizer_reset = False
        elif self.cur_stage == self.STAGE_VALID:
            self.model.eval()
            self.cur_dataloader = self.dataloader.valid_loader
        elif self.cur_stage == self.STAGE_TEST:
            self.model.eval()
            self.cur_dataloader = self.dataloader.test_loader
        self.setup_logging_before_stage()

    def load_batch_pack(self):
        """ this method is used as an iterator of the current dataloader, which
        also loads the data from CPU to GPU.
        """
        for batch in self.cur_dataloader:
            # should return a dict of result to use
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=self.is_parallel)
            batch_pack = BatchPack(gt=batch)
            yield batch_pack

    def train(self, batch_pack: BatchPack) -> BatchPack:
        """ this method completes the training with a mini-batch

        :param batch_pack: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of training result for the mini-batch
        """
        # should return a dict of result to use
        self.optimizer.zero_grad()
        self.model_forward_backward(batch_pack, backward=True)
        self.optimizer.step()
        return batch_pack

    def _train(self) -> BatchPack:
        """ this private method has better not be changed. It is designed to
        finish the training over the training dataset and log the result
        properly.

        :return: BatchPack of training result of the last mini-batch
        """
        self.cur_stage = self.STAGE_TRAIN
        self.setup_before_stage()

        with torch.enable_grad():
            for batch_pack in self.load_batch_pack():
                self.train(batch_pack)
                synchronize()
                self.update_logging_in_stage(batch_pack)
                synchronize()
        return batch_pack

    def valid(self, batch_pack: BatchPack) -> BatchPack:
        """ this method completes the validation with a mini-batch

        :param batch_pack: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of validation result for the mini-batch
        """
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch_pack)

    def _valid(self) -> BatchPack:
        """ this private method has better not be changed. It is designed to
        finish the validation over the validation dataset and log the result
        properly.

        :return: dict of validation result of the last mini-batch
        """
        self.cur_stage = self.STAGE_VALID
        self.setup_before_stage()

        with torch.no_grad():
            for batch_pack in self.load_batch_pack():
                self.valid(batch_pack)
                synchronize()
                self.update_logging_in_stage(batch_pack)
                synchronize()
        return batch_pack

    def test(self, batch_pack: BatchPack) -> BatchPack:
        """ this method completes the test with a mini-batch

        :param batch_pack: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of test result for the mini-batch
        """
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch_pack)

    def _test(self) -> BatchPack:
        """ this private method has better not be changed. It is designed to
        finish the test over the test dataset and log the result properly.

        :return: BatchPack of test result of the last mini-batch
        """
        self.cur_stage = self.STAGE_TEST
        self.setup_before_stage()

        with torch.no_grad():
            for batch_pack in self.load_batch_pack():
                self.test(batch_pack)
                synchronize()
                self.update_logging_in_stage(batch_pack)
                synchronize()
        return batch_pack

    def setup_logging_before_stage(self):
        """ setup logging before start to train/validate/test the model. For
        example, update the progress bar.
        """
        if self.is_rank0:
            self.progress_bars[self.cur_stage].reset(len(self.cur_dataloader))
            self.rank0_setup_logging_before_stage()

    def update_logging_in_stage(self, result: BatchPack):
        """ log the result during training/validation/testing, including
        recording the loss with `self.meter`, and call the rank0 process to do
        visualization.

        :param result: BatchPack instance
        """
        if isinstance(result.loss, dict):
            for k, v in result.loss.items():
                if v is not None:
                    key = f'{self.cur_stage}/{k}-loss'
                    self.meter.record(
                        tag=key, value=v.item(),
                        weight=result.batch_size,
                        record_op=Meter.RecordOp.APPEND,
                        reduce_op=Meter.ReduceOp.SUM
                    )
                    self.in_stage_meter_keys.add(key)
        else:
            key = f'{self.cur_stage}/loss'
            self.meter.record(
                tag=key, value=result.loss.item(),
                weight=result.batch_size,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.SUM
            )
            self.in_stage_meter_keys.add(key)
        if self.is_rank0:
            self.rank0_update_logging_in_stage(result)

    def summarize_logging_after_stage(self) -> OrderedDict:
        """ get the summary of the result over the whole training/validation/
        test dataset.

        :return: dict of stage summary
        """
        summary = OrderedDict()

        if self.tboard is None:
            summary['name'] = self.option.name
            summary['datetime'] = self.option.datetime
            summary['epoch'] = self.epoch
            if self.option.model.pth_path is None:
                summary['pth_file'] = 'None'
            else:
                summary['pth_file'] = os.path.basename(
                    self.option.model.pth_path)

        for key in sorted(list(self.in_stage_meter_keys)):
            if key.startswith(self.cur_stage):
                summary[key] = self.meter.mean(key)

        for k, v in summary.items():
            tag = f'epoch-{k}'
            self.meter.record(
                tag=tag, value=v,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.STORE
            )
            info(__name__, f'{tag} = {v}')

        if self.is_rank0:
            self.rank0_update_logging_after_stage(summary)
        return summary

    def rank0_setup_logging_before_stage(self):
        """ this method should do some setups only needed on the rank0 process,
        such as tensorboard logging, before a stage begins.
        """
        pass

    def rank0_update_logging_in_stage(self, result: BatchPack):
        """ this method should update logging only needed on the rank0 process,
        such as tensorboard logging, during a stage.

        :param result: BatchPack that stores ground truth, prediction, loss
            and batch size
        """
        self.progress_bars[self.cur_stage].update()
        self.progress_bars[self.cur_stage].refresh()
        if self.tboard is not None:
            if isinstance(result.loss, dict):
                for k, v in result.loss.items():
                    self.tboard.add_scalar(
                        f'batch-{self.cur_stage}/{k}-loss', v,
                        self.batch_cnt[self.cur_stage]
                    )
                    self.tboard.add_scalar(
                        f'batch/{k}', v, self.batch_cnt['all']
                    )
            else:
                self.tboard.add_scalar(
                    f'batch-{self.cur_stage}/loss', result.loss.item(),
                    self.batch_cnt[self.cur_stage]
                )
                self.tboard.add_scalar(
                    f'batch/loss', result.loss.item(), self.batch_cnt['all']
                )
            self.batch_cnt[self.cur_stage] += 1
            self.batch_cnt['all'] += 1

    def rank0_update_logging_after_stage(self, summary: dict):
        """ this method should update logging only needed on the rank0
        process, such as tensorboard logging, after a stage

        :param summary: dict of stage summary
        :return:
        """
        if self.tboard is not None:
            for k, v in summary.items():
                self.tboard.add_scalar(f'epoch-{k}', v, self.epoch)

    def rank0_update_logging_after_epoch(self):
        """ this method should update logging only needed on the rank0 process.
        """
        pass
