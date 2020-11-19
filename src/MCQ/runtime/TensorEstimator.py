from typing import Tuple, Union
from logging import Logger

import torch
from torch import nn
import numpy as np

from MCQ import Consts
from MCQ.base import Restorable


# From TF Agents tensor_normalizer.py
# https://github.com/tensorflow/agents/blob/master/tf_agents/utils/tensor_normalizer.py
class TensorEstimator(Restorable):
    def __init__(self, logger: Logger = None):
        super().__init__()
        self.logger = logger or Consts.Logger
        self._loggedTensors = dict()
        self._mean = dict()
        self._variance = dict()


class RunningMeanVar(TensorEstimator):
    def __init__(self, momentum=0.99):
        super().__init__()
        self._updateRate = 1 - momentum

    def Estimate(self, tensorWithName: Tuple[str, torch.Tensor], outerDims: Tuple[int]):
        name, tensor = tensorWithName
        name = "estimate/" + name
        mean = tensor.mean(outerDims, keepdim=True)
        if name not in self._loggedTensors:
            self._loggedTensors[name] = { "runningMean": torch.zeros_like(mean), "runningVar": torch.ones_like(mean) }
        runningMean = self._loggedTensors[name]["runningMean"]
        runningVar = self._loggedTensors[name]["runningVar"]
        var = ((tensor - runningMean) ** 2).mean(outerDims, keepdim=True)
        self._loggedTensors[name]["runningMean"] += self._updateRate * (mean - runningMean)
        self._loggedTensors[name]["runningVar"] += self._updateRate * (var - runningVar)
        self._mean[name] = self._loggedTensors[name]["runningMean"]
        self._variance[name] = self._loggedTensors[name]["runningVar"]
        return self._loggedTensors[name]["runningMean"], self._loggedTensors[name]["runningVar"]


class StreamingMeanVar(TensorEstimator):
    def Estimate(self, tensorWithName: Tuple[str, Union[torch.Tensor, np.ndarray]], outerDims: Tuple[int]):
        name, tensor = tensorWithName
        name = "estimate/" + name
        count = 1
        for d in outerDims:
            count *= tensor.shape[d]
        if name not in self._loggedTensors:
            self._loggedTensors[name] = {"count": 1e-8, "meanSum": 0.0, "varSum": 0.0}
        meanSum = self._loggedTensors[name]["meanSum"]
        varSum = self._loggedTensors[name]["varSum"]
        historyCount = self._loggedTensors[name]["count"]
        if isinstance(tensor, torch.Tensor):
            varSum += torch.sum((tensor - meanSum / historyCount) ** 2, outerDims, keepdim=True)
            meanSum += torch.sum(tensor, outerDims, keepdim=True)
        else:
            varSum += ((tensor - meanSum / historyCount) ** 2).sum(outerDims, keepdims=True)
            meanSum += tensor.sum(outerDims, keepdims=True)
        self._loggedTensors[name]["count"] += count
        self._loggedTensors[name]["meanSum"] = meanSum
        self._loggedTensors[name]["varSum"] = varSum
        meanEst = self._loggedTensors[name]["meanSum"] / self._loggedTensors[name]["count"]
        varEst = self._loggedTensors[name]["varSum"] / self._loggedTensors[name]["count"]

        self._mean[name] = meanEst
        self._variance[name] = varEst
        return meanEst, varEst
