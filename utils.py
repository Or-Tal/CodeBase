import os
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
            _log_obj(k, v, prefix + "  ")
    else:
        _logger.info(f"{prefix}{name}: {obj}")


def log_args(args, _logger):
    _log_obj("Args", vars(args)["_content"], "")


def init_wandb(args):
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    wandb.init(mode=wandb_mode, project=args.wandb.project, entity=args.wandb.wandb_entity,
               config=args, group=args.experiment_name,
               resume=not args.reset, name=args.experiment_name)