# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import logging
import os
import hydra
import torch

from solvers.solver_factory import SolverFactory
from utilities import distributed
from utilities.distributed import start_ddp_workers
from utils import init_wandb, log_args

logger = logging.getLogger(__name__)


def log_arguments(args):
    return log_args(args, logger)


def init_hydra_and_logs(args):
    global __file__
    log_arguments(args)
    for key, value in args.data.items():
        if isinstance(value, str) and key not in ["matching"] and ".json" in value:
            args.data[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.training.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)


def _main(args):

    # init wandb
    init_wandb(args)

    # init hydra and logs
    init_hydra_and_logs(args)

    if args.ddp and args.local_rank is None:
        start_ddp_workers(args)
    else:
        init_train_loop(args)


def init_train_loop(args):
    # init seed and distrib
    torch.manual_seed(args.seed)
    distributed.init(args)

    # initialize a solver object
    solver = SolverFactory.get_solver(args)

    # run training
    solver.train()


@hydra.main(config_path="configurations", config_name="main_config")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
