# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import logging

import torch
import os
import subprocess as sp
import sys
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from hydra import utils as hutils


local_rank = 0
num_gpus = 1


def init(configuration):
    """
    initialize a distributed run using temporary randezvous file
    """
    # use these variables to update the different executions
    global local_rank, num_gpus

    # init distrib process
    if configuration.ddp:
        assert configuration.local_rank is not None and configuration.num_gpus is not None
        local_rank = configuration.local_rank
        num_gpus = configuration.num_gpus
    if num_gpus == 1:
        return

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=configuration.ddp_backend,
        init_method='file://' + os.path.abspath(configuration.rendezvous_file),
        world_size=num_gpus,
        rank=local_rank)


def average_over_all_gpus(values):
    """
    Averages collected values on a distributed run

    :param values: values to average over, should be a 1D float32 vector.
    average.
    Average all the relevant metrices across processes
    `metrics`should be a 1D float32 vector. Returns the average of `metrics`
    over all hosts. You can use `count` to control the weight of each worker.
    """
    if num_gpus == 1:
        return values
    tensor = torch.tensor(list(values) + [1], device='cuda', dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy()


def wrap(model):
    """
    Wrap a model for distributed training.
    """
    if torch.cuda.is_available():
        model.to('cuda')

    if num_gpus == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())


def barrier():
    if num_gpus > 1:
        torch.distributed.barrier()


def get_loader(dataset, *args, shuffle=False, custom_dloader_class=DataLoader, **kwargs):
    """loader.
    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.
    :param dataset: the dataset to be parallelized
    :param args: relevant args for the loader
    :param shuffle: shuffle examples
    :param klass: loader class
    :param kwargs: relevant args
    """

    if num_gpus == 1:
        return custom_dloader_class(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return custom_dloader_class(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(local_rank, len(dataset), num_gpus)))
        return custom_dloader_class(dataset, *args, shuffle=shuffle)


class ChildrenManager:
    def __init__(self, logger=None):
        self.children = []
        self.failed = False
        self.logger = logger if logger else logging.getLogger(__name__)

    def add(self, child):
        child.rank = len(self.children)
        self.children.append(child)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self.logger.error("An exception happened while starting workers %r", exc_value)
            self.failed = True
        try:
            while self.children and not self.failed:
                for child in list(self.children):
                    try:
                        exitcode = child.wait(0.1)
                    except sp.TimeoutExpired:
                        continue
                    else:
                        self.children.remove(child)
                        if exitcode:
                            self.logger.error(f"Worker {child.rank} died, killing all workers")
                            self.failed = True
        except KeyboardInterrupt:
            self.logger.error("Received keyboard interrupt, trying to kill all workers.")
            self.failed = True
        for child in self.children:
            child.terminate()
        if not self.failed:
            self.logger.info("All workers completed successfully")


def start_ddp_workers(cfg, logger=None):
    import torch as th
    if logger is None:
        logger = logging.getLogger(__name__)
    # log = cfg.hydra.job_logging.handlers.file.filename
    log = hutils.HydraConfig().cfg.hydra.job_logging.handlers.file.filename
    rendezvous_file = Path(cfg.rendezvous_file)
    if rendezvous_file.exists():
        rendezvous_file.unlink()

    num_gpus = th.cuda.device_count()
    if not num_gpus:
        logger.error(
            "DDP is only available on GPU. Make sure GPUs are properly configured with cuda.")
        sys.exit(1)
    logger.info(f"Starting {num_gpus} worker processes for DDP.")
    with ChildrenManager() as manager:
        for local_rank in range(num_gpus):
            kwargs = {}
            argv = list(sys.argv)
            argv += [f"num_gpus={num_gpus}", f"local_rank={local_rank}"]
            if local_rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = sp.DEVNULL
                kwargs['stderr'] = sp.DEVNULL
                log += f".{local_rank}"
                argv.append("hydra.job_logging.handlers.file.filename=" + log)
            manager.add(sp.Popen([sys.executable] + argv, cwd=hutils.get_original_cwd(), **kwargs))
    sys.exit(int(manager.failed))