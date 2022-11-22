# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.

from abc import ABC, abstractmethod
from typing import Tuple, Union, Any
from torch.utils.data.dataloader import DataLoader


class BaseBuilder(ABC):

    @staticmethod
    @abstractmethod
    def get_tr_cv_tt_loaders(dset_config) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        """
        this method should receive a dictionary of arguments and return train, valid, and test data loaders
        Optional: the 4th argument could be an additional dataloader used to generate samples at the end of
        each evaluation step; if not given - test dataloader would be used.
        """
        raise NotImplementedError()

    @staticmethod
    def create_dataloader(dset_config, dataset, batch_size=None, shuffle=False):
        return DataLoader(dataset=dataset, batch_size=batch_size or dset_config.batch_size,
                          num_workers=dset_config.num_workers or 1, shuffle=shuffle)
