import copy
from collections import OrderedDict
import types

import torch
from torch import nn

def Update(self, newModel):
    with torch.no_grad():
        modelParams = OrderedDict(self.named_parameters())
        shadowParams = OrderedDict(newModel.named_parameters())

        # check if both model contains the same set of keys
        assert modelParams.keys() == shadowParams.keys()

        for name, param in modelParams.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadowParams[name].sub_((1. - self._decay) * (shadowParams[name] - param))

        model_buffers = OrderedDict(self.named_buffers())
        shadow_buffers = OrderedDict(self.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

def ExponentialMovingAvg(model: nn.Module, decay: float):
    shadowModel = copy.deepcopy(model)
    shadowModel._decay = decay
    shadowModel.Update = types.MethodType(Update, shadowModel)
    return shadowModel