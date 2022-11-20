# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import torch.nn.functional as F


class LossFactory:
    LOSS_FUNCTIONS = {
        "l1": F.l1_loss
    }

    @staticmethod
    def get_loss_func(loss_name):
        assert loss_name.lower() in LossFactory.LOSS_FUNCTIONS.keys(), "unsupported loss function was given"
        return LossFactory.LOSS_FUNCTIONS[loss_name.lower()]
