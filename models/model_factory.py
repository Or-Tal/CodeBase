# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from models.dummy_ae import DummyAE
from utilities import distributed


class ModelFactory:

    allowed_models = {
        "dummy": DummyAE
    }

    @staticmethod
    def wrap_model(model):
        if isinstance(model, list):
            model = [distributed.wrap(m) for m in model]
            return model
        return distributed.wrap(model)

    @staticmethod
    def get_model(model_args):
        assert hasattr(model_args, "model_class_name"), f"Please add 'model_class_name' to model configuration.\n{model_args}"
        if model_args.model_class_name.lower() not in ModelFactory.allowed_models.keys():
            raise ValueError("given model_class_name is not supported by model factory.")
        return ModelFactory.wrap_model(ModelFactory.allowed_models[model_args.model_class_name.lower()](model_args))
