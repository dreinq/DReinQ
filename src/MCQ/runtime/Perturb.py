import torch
from torch import nn

class Perturber:
    def __init__(self):
        pass

    def Perturb(self, model: nn.Module):
        stateDict = model.state_dict()
        def _perturb(item):
            if isinstance(item, dict):
                for key, value in stateDict.items():
                    stateDict[key] = _perturb(value)
            elif isinstance(item, list):
                return item.__class__(_perturb(x) for x in item)
            else:
                ...