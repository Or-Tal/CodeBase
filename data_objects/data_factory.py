# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from typing import Tuple, Union
from torch.utils.data.dataloader import DataLoader
from data_objects.dummy_builder import DummyBuilder


class DataFactory:

    valid_builders = {
        "dummy": DummyBuilder
    }

    @staticmethod
    def get_loaders(data_config) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        assert hasattr(data_config, "data_builder_name" ), "Please add 'data_builder_name' to model configuration."
        if data_config.data_builder_name.lower() not in DataFactory.valid_builders.keys():
            raise ValueError(f"DsetBuilder: {data_config.data_builder_name} is not supported by DataFactory."
                             f"\nPlease make sure implementation is valid.")
        else:
            return DataFactory.valid_builders[data_config.data_builder_name].get_tr_cv_tt_loaders(data_config)