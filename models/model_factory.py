# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from models.dummy_ae import DummyAE


class ModelFactory:

    allowed_models = {
        "dummy": DummyAE
    }

    @staticmethod
    def get_model(model_args):
        assert hasattr("model_class_name", model_args), "Please add 'model_class_name' to model configuration."
        if model_args.model_class_name.lower() not in ModelFactory.allowed_models.keys():
            raise ValueError("given model_class_name is not supported by model factory.")
        return DummyAE(model_args)
