
import os
import json
import functools
from typing import Dict, Type, Union, Tuple, Callable

import torch

from MCQ import Consts


def _replacePlaceholderWithRealValue(jsonPath: str, args):
    formatMap = {
        "data": Consts.DataDir,
        "self": os.path.dirname(jsonPath)
    }
    if isinstance(args, str):
        return args.format(**formatMap)
    elif isinstance(args, dict):
        return {
            key: _replacePlaceholderWithRealValue(jsonPath, value)
            for key, value in args.items()
        }
    elif isinstance(args, (list, set, tuple)):
        return args.__class__(
            _replacePlaceholderWithRealValue(jsonPath, v) for v in args)
    else:
        return args


class ConfigBase:
    def Update(self, **args):
        for key, value in self.__dict__.items():
            if key in args:
                new = args[key]
                if isinstance(value, ConfigBase):
                    setattr(self, key, value.Update(**new))
                elif isinstance(value, dict):
                    if key in _configMap:
                        setattr(
                            self, key, {
                                k: _configMap[key][0]().Update(**v)
                                for k, v in new.items()
                            })
                    else:
                        setattr(self, key, new)
                elif isinstance(value, (list, set, tuple)):
                    if key in _configMap:
                        setattr(
                            self, key,
                            value.__class__(_configMap[key][0]().Update(**v)
                                            for v in new))
                    else:
                        setattr(self, key, value.__class__(v for v in new))
                elif key in _configMap:
                    hintType = _configMap[key]
                    if isinstance(hintType, dict):
                        setattr(self, key, {
                            k: hintType[0]().Update(**v)
                            for k, v in new.items()
                        })
                    elif isinstance(hintType, (list, set, tuple)):
                        setattr(
                            self, key,
                            value.__class__(hintType[0]().Update(**v)
                                            for v in new))
                    elif isinstance(hintType, type(ConfigBase)):
                        setattr(self, key, hintType().Update(**new))
                    else:
                        setattr(self, key, new)
                else:
                    setattr(self, key, new)
        return self

    def __eq__(self, other):
        if isinstance(other, ConfigBase):
            return str(self) == str(other)
        return False

    def __repr__(self):
        return json.dumps(self, default=Serialize)

    def __str__(self):
        return json.dumps(self, default=Serialize, indent=4)


def Serialize(obj):
    if isinstance(obj, ConfigBase):
        result = {}
        for key, value in obj.__dict__.items():
            if value is None:
                continue
            result[key] = Serialize(value)
        return result
    elif isinstance(obj, dict):
        new = obj.__class__()
        for k, v in obj.items():
            new[k] = Serialize(v)
        return new
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(Serialize(v) for v in obj)
        return new
    else:
        return obj


class SchedulerConfig(ConfigBase):
    @property
    def Type(self) -> str:
        return self.type

    @property
    def Args(self) -> Dict[str, object]:
        return self.args

    def __init__(self):
        self.type: str = None
        self.args: Dict[str, object] = dict()

class OptimizerConfig(ConfigBase):
    @property
    def Type(self) -> str:
        return self.type

    @property
    def Args(self) -> Dict[str, object]:
        return self.args

    @property
    def Scheduler(self) -> SchedulerConfig:
        return self.scheduler

    def __init__(self):
        self.type: str = None
        self.args: Dict[str, object] = dict()
        self.scheduler: SchedulerConfig = SchedulerConfig()


class HParamsConfig(ConfigBase):

    def __init__(self):
        self.m: int = 4
        self.k: int = 256
        self.batchSize: int = 100
        self.eps: float = 0.2
        self.alpha: float = 0.05
        self.alphaDiscount = 0.999
        self.gamma: float = 0.99
        self._lambda: float = 0.95
        self.gradNorm: float = 0.5
        self.normalizeObs: bool = True
        self.normalizeRew: bool = True
        self.networkArgs: Dict[str, object] = dict()

    def Update(self, **args):
        super().Update(**args)
        assert self.m > 0
        assert (self.k & (self.k - 1) == 0) and self.k != 0
        return self

    @property
    def BatchSize(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.batchSize

    @BatchSize.setter
    def BatchSize(self, value):
        self.batchSize = value

    @property
    def Eps(self) -> float:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.eps

    @property
    def Alpha(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.alpha

    @property
    def AlphaDiscount(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.alphaDiscount

    @property
    def Gamma(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.gamma

    @property
    def Lambda(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self._lambda

    @property
    def GradNorm(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.gradNorm

    @property
    def M(self) -> int:
        """The number of sub-codebooks in quantization, L = m * log2(k).

        Returns:
            int -- sub-codebooks
        """
        return self.m

    @property
    def K(self) -> int:
        """The number of codewords in a sub-codebook, must be power of 2, L = m * log2(k).

        Returns:
            int -- codewords
        """
        return self.k

    @property
    def NormalizeObs(self) -> bool:
        return self.normalizeObs

    @property
    def NormalizeRew(self) -> bool:
        return self.normalizeRew

    @property
    def NetworkArgs(self) -> Dict[str, object]:
        return self.networkArgs

    @property
    def ParamsForActorCritic(self):
        params = {
            "m": self.m,
            "k": self.k
        }
        params.update(self.networkArgs)
        return params


class Config(ConfigBase):
    def __init__(self):
        self.hParams: HParamsConfig = HParamsConfig()
        self.evalStep: int = 200
        self.method = "PPO"
        self.port: int = 12345
        self.epoch: int = -1
        self.useVRam: int = -1
        self.gpus: int = 1
        self.wantsMore: bool = False
        self.dataset: str = None
        self.optimizers: Dict[str, OptimizerConfig] = {
            "network": OptimizerConfig()
        }

    @staticmethod
    def Read(configPath: str):
        with open(configPath, "r") as fp:
            args = json.load(fp)
        args = _replacePlaceholderWithRealValue(configPath, args)
        return Config().Update(**args)

    def Update(self, **args):
        super().Update(**args)
        assert self.port > 0 and self.port < 65536
        return self

    @property
    def OptimAndSchFns(self) -> Dict[str, Tuple[Callable, Union[Callable, None]]]:
        return OptimizerFnsFromDict(self.optimizers)

    @property
    def HParams(self) -> HParamsConfig:
        return self.hParams

    @property
    def Optimizers(self) -> Dict[str, OptimizerConfig]:
        return self.optimizers

    @Optimizers.setter
    def Optimizers(self, value):
        self.optimizers = value

    @property
    def Port(self) -> int:
        return self.port

    @property
    def Dataset(self) -> str:
        return self.dataset

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

    @property
    def UseVRam(self) -> int:
        return self.useVRam

    @property
    def Method(self) -> str:
        return self.method

    @property
    def EvalStep(self) -> int:
        return self.evalStep


_configMap: Dict[str, Type] = {
    "optimizers": {
        0: OptimizerConfig
    },
    "hParams": HParamsConfig,
    "scheduler": SchedulerConfig
}

_optimMap: Dict[str, Type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

_lrMap: Dict[str, Type[torch.optim.lr_scheduler.Optimizer]] = {
    "exponentialDecay": torch.optim.lr_scheduler.ExponentialLR
}


def OptimizerFnsFromDict(optimDict: Dict[str, OptimizerConfig]) -> Dict[str, Tuple[Callable, Union[Callable, None]]]:
    return {key: _optimizerFnFromConfig(value) for key, value in optimDict.items()}


def _optimizerFnFromConfig(optimConfig: OptimizerConfig):
    optimFn = functools.partial(_optimMap[optimConfig.Type], **optimConfig.Args)
    if optimConfig.Scheduler.Type is not None:
        schedulerFn = functools.partial(_lrMap[optimConfig.Scheduler.Type], **optimConfig.Scheduler.Args)
    else:
        schedulerFn = None
    return (optimFn, schedulerFn)
