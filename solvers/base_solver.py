# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import itertools
import os
import time
from typing import Union, List
from pathlib import Path
from accelerate.accelerator import Accelerator
from data_objects.data_factory import DataFactory
from models.model_factory import ModelFactory
import torch
import torch.nn as nn
import logging
import tqdm
import wandb
from solvers.loss_function_factory import LossFactory
from utils import copy_state

DEF_SUPPORTED_OPTIMIZERS = {
    "adam": torch.optim.Adam
}

logger = logging.getLogger(__name__)


class BaseSolver:

    def build_all_data_loaders(self, args):
        """
        This should return all data loaders for the training scheme, default assumes:
        [Train_dloader, Validation_dloader, Test_dloader, Enhance_dloader (default is None)]
        """
        return DataFactory.get_loaders(args.data)

    def initialize_models(self, args) -> Union[nn.Module, List[nn.Module]]:
        """
        This should return a model object / a list of model objects
        """
        return ModelFactory.get_model(args.model)

    def get_single_optimizer(self, training_config, models):
        """
        this function initializes a single optimizer for given model / models list
        """
        assert str(
            training_config.optim).lower() in DEF_SUPPORTED_OPTIMIZERS.keys(), "unsupported optimizer was given - see base_builder.py for supported dict of optimizers"
        assert isinstance(models, list) or isinstance(models, nn.Module)
        if isinstance(models, list):
            params = itertools.chain(*[m.parameters() for m in models])
        else:
            params = models.parameters()

        return DEF_SUPPORTED_OPTIMIZERS[training_config.optim](params, **training_config.optimizer_args)

    def get_optimizers(self, training_config, models):
        """
        This should return all optimizers needed for training
        """
        if isinstance(models, nn.Module):
            return self.get_single_optimizer(training_config, models)
        elif isinstance(models, list):
            return [self.get_single_optimizer(training_config, m) for m in models]

    @staticmethod
    def _flatten_list_for_accellerator(*object_list):
        lengths = []
        flattened_objects = []
        for o in object_list:
            if isinstance(o, list):
                lengths.append(len(o))
                for o2 in o:
                    flattened_objects.append(o2)
            else:
                lengths.append(1)
                flattened_objects.append(o)
        return flattened_objects, lengths

    @staticmethod
    def _unflatten_list_by_length(flatten_object_list, lengths):
        out = []
        i = 0
        for le in lengths:
            out.append(flatten_object_list[i: i + le])
            i += le
        return out

    def initialize_with_accellerator(self, args, *object_list, device_placement=None):
        """
        This function uses Accelerator Class by huggingface to manage all distributed training.
        Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
        order.
        :param args: full configuration
        :param object_list: Any of the following objects:
                                - `torch.utils.data.DataLoader`: PyTorch Dataloader
                                - `torch.nn.Module`: PyTorch Module
                                - `torch.optim.Optimizer`: PyTorch Optimizer
                                - `torch.optim.lr_scheduler._LRScheduler`: PyTorch LR Scheduler
        :param device_placement: (`List[bool]`, *optional*):
                Used to customize whether automatic device placement should be performed for each object passed. Needs
                to be a list of the same length as `args`.
        :return: accelerator, list of objects as passed
        """
        accelerator = Accelerator()
        flattened_objects, lengths = self._flatten_list_for_accellerator(object_list)
        flattened_objects = accelerator.prepare(*flattened_objects)
        return accelerator, self._unflatten_list_by_length(flattened_objects, lengths)

    def initialize_all_training_objects(self, args):
        """
        this function returns a list of all objects required for the training process
        :param args:
        :return:
        """
        models = ModelFactory.get_model(args.model)
        data_loaders = DataFactory.get_loaders(args.data)
        optimizers = self.get_optimizers(args.training, models)

        accelerator, tmp = \
            self.initialize_with_accellerator(args, [models, data_loaders, optimizers])

        # TODO: debug why this phenomena happens?
        while (isinstance(tmp[0], tuple) or isinstance(tmp[0], list)) and len(tmp) == 1:
            tmp = tmp[0]

        logger.info(f"len: {len(tmp)} | [{', '.join([f'{type(t)}' for t in tmp])}]")

        models, data_loaders, optimizers = tmp[0], tmp[1], tmp[2]

        return accelerator, models, data_loaders, optimizers

    def define_all_objects(self, args):
        accelerator, models, data_loaders, optimizers = self.initialize_all_training_objects(args)
        self.accelerator = accelerator
        self.model = models
        self.tr_dl, self.cv_dl, self.tt_dl, self.enh_dl = data_loaders
        self.opt = optimizers

    def set_state_dict_models(self, state):
        self.model.load_state_dict(state)

    def get_state_dict_models(self):
        return self.model.state_dict()

    def set_state_dict_optimizers(self, state):
        self.opt.load_state_dict(state)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def reset_models(self, args, checkpoint_content):
        # continue best or last
        if args.training.restart:
            return
        state_key = "best_state" if args.training.continue_best else "last_state"
        self.set_state_dict_models(checkpoint_content['model'][state_key])

    def reset_optimizers(self, args, checkpoint_content):
        if args.training.restart:
            return
        self.set_state_dict_optimizers(checkpoint_content["optimizer"])

    def reset_history(self, args, checkpoint_content):
        if args.training.restart:
            return []
        return checkpoint_content["history"]

    def reset(self, args):
        if self.checkpoint_file_path.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file_path}')
            checkpoint_content = torch.load(self.checkpoint_file_path, 'cpu')
            self.reset_optimizers(args, checkpoint_content)
            self.reset_models(args, checkpoint_content)
            self.history = self.reset_history(args, checkpoint_content)

    def serialize_models(self):
        return {"best": self.best_state, "last": copy_state(self.model.state_dict())}

    def serialize_optimizers(self):
        return copy_state(self.opt.state_dict())

    def define_checkpoint_for_serialization(self):
        checkpoint = {
            'model': self.serialize_models(),
            'optimizer': self.serialize_optimizers(),
            'history': self.history,
            'args': self.args
        }
        return checkpoint

    def serialize(self):
        checkpoint = self.define_checkpoint_for_serialization()
        tmp_path = str(self.checkpoint_file_path) + ".tmp"
        torch.save(checkpoint, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file_path)

    def __init__(self, args):
        self.args = args
        self.define_all_objects(args)

        # checkpoint related
        self.checkpoint_file_path = Path(args.training.checkpoint_file)
        self.history_file_path = Path(args.training.history_file)

        # training related
        self.history = list()
        self.best_state = None
        self.reset(args)

    def accumulate_loss(self, losses, validation):
        return torch.mean(losses)

    def inference(self, input_signal):
        return self.model(input_signal)

    def run_single_epoch(self, loss_function, epoch_num, validation=False):
        loader = self.cv_dl if validation else self.tr_dl
        losses = []
        for batch in tqdm.tqdm(loader, desc=f"Epoch {epoch_num} [{'Valid' if validation else 'Train'}]:", leave=False):
            losses.append(self.run_single_batch(loss_function, batch, epoch_num, validation))
        return self.accumulate_loss(losses, validation)

    def optimize(self, loss):
        self.accelerator.backward(loss)
        self.opt.step()
        self.opt.zero_grad()

    def run_single_batch(self, loss_function, batch, epoch_num, validation=False):
        if validation:
            with torch.no_grad():
                outputs = self.model(batch)
        else:
            outputs = self.model(batch)
        loss = loss_function(outputs)
        if not validation:
            self.optimize(loss)
        return loss.item()

    def get_best_loss_from_history(self):
        min_loss = 1e20
        for step in self.history:
            if step[0] < min_loss:
                min_loss = step[0]
        return min_loss

    def check_if_model_has_improved(self, loss, best_loss):
        return best_loss > loss

    def update_history(self, valid_loss, metrics):
        self.history.append(valid_loss)

    def evaluate(self, eval_over_test_set=False):
        # TODO implement evaluation over desired metrics
        return

    def log_to_wandb(self, epoch_num, train_loss, valid_loss, metrics, test_metrics=None):
        wandb.log(
            {
                "Train loss": train_loss,
                "Validation loss": valid_loss,
            },
            step=epoch_num
        )

    def log_audio_files_to_wandb(self, epoch_num):
        if not self.enh_dl:
            return
        with torch.no_grad():
            for signal, filename in self.enh_dl:
                wandb.log({
                    f"test samples/{filename}/audio": wandb.Audio(self.inference(signal).cpu().squeeze().numpy(),
                                                                  sample_rate=self.args.data.sample_rate,
                                                                  caption=filename)
                }, step=epoch_num
                )

    def log_summary_for_epoch(self, epoch_num, train_time, val_time, train_loss, val_loss, val_metrics):
        logger.info(
            f"Epoch: {epoch_num}| Tr_loss [{train_time:.2} Sec] : {train_loss:f5} | Val_loss [{val_time:.2} Sec]: {val_loss:.f5}")

    def train(self):
        if self.history:
            logger.info("Restored training. Previous iterations:")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
                logger.info(f"Epoch {epoch + 1}: {info}")
        if len(self.history) < self.args.training.epochs:
            logger.info(f"==== TRAINING ====")

        # Training loop
        best_loss = self.get_best_loss_from_history()
        loss_func = LossFactory.get_loss_func(self.args.training.loss_name)
        for epoch in range(len(self.history), self.args.training.epochs):
            # train single epoch
            self.set_train()
            start_time = time.time()
            loss = self.run_single_epoch(loss_func, epoch, validation=False)
            end_time = time.time()

            self.set_eval()
            valid_start_time = time.time()
            val_loss = self.run_single_epoch(loss_func, epoch, validation=True)
            metrics = self.evaluate()
            valid_end_time = time.time()

            if self.args.training.eval_over_test_set_interval > 0 and \
                    epoch % self.args.training.eval_over_test_set_interval == \
                    self.args.training.eval_over_test_set_interval - 1:
                test_metrics = self.evaluate(eval_over_test_set=True)
                self.log_audio_files_to_wandb(epoch + 1)
            else:
                test_metrics = None

            # log
            self.log_summary_for_epoch(epoch + 1, end_time - start_time, valid_end_time - valid_start_time,
                                       loss, val_loss, metrics)
            self.log_to_wandb(epoch + 1, loss, val_loss, metrics, test_metrics)

            # update history
            self.update_history(val_loss, metrics)

            # swap best state with current state
            if self.check_if_model_has_improved(val_loss, best_loss):
                self.best_state = self.get_state_dict_models()

            # serialize
            if epoch % self.args.training.serialization_interval == 0:
                self.serialize()

        # Evaluate over
