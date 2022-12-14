# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import os
from contextlib import contextmanager

import torch
import wandb as wandb


def _log_obj(name, obj, prefix, _logger):
    if name in ["wandb", "dset", "model"]:
        try:
            obj = vars(obj)["_content"]
        except Exception:
            return
    if isinstance(obj, dict):
        _logger.info(f"{prefix}{name}:")
        for k, v in obj.items():
            _log_obj(k, v, prefix + "  ", _logger)
    else:
        _logger.info(f"{prefix}{name}: {obj}")


def parse_value(value):
    if isinstance(value, dict):
        return copy_state(value)
    elif isinstance(value, list):
        return [parse_value(v) for v in value]
    elif 'torch' in f"{type(value)}":
        return value.cpu().clone()
    return value


def copy_state(state):
    return {k: parse_value(v) for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:
        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def log_args(args, _logger):
    _log_obj("Args", vars(args)["_content"], "", _logger)


def init_wandb(args):
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.training.wandb.mode
    wandb.init(mode=wandb_mode, project=args.training.wandb.project, entity=args.training.wandb.wandb_entity,
               config=args, group=args.experiment_name,
               resume=not args.training.restart, name=args.experiment_name)
