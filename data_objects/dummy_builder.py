# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from typing import Any, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from data_objects.base_builder import BaseBuilder
from data_objects.simple_audio_dataset import SimpleAudioDataset


class DummyBuilder(BaseBuilder):

    @staticmethod
    def get_tr_cv_tt_loaders(data_config) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        tr_dset = SimpleAudioDataset(data_config.data, "tr")
        cv_dset = SimpleAudioDataset(data_config.data, "cv", ignore_length=True)
        tt_dset = SimpleAudioDataset(data_config.data, "tt", ignore_length=True, include_path=True)

        return (
            BaseBuilder.create_dataloader(data_config, tr_dset, shuffle=True),
            BaseBuilder.create_dataloader(data_config, cv_dset, batch_size=1, shuffle=False),
            BaseBuilder.create_dataloader(data_config, tt_dset, batch_size=1, shuffle=False),
            None)
