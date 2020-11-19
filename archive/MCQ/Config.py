
import os, json
from typing import Dict, Type, List, Union, Callable, Tuple

import tensorflow as tf

from MCQ import Consts
from MCQ.optimizers import GradientAccumulatedAdam


def replace_placeholder_with_real_value(json_path: str, args):
    format_map = {
        "data": Consts.DataDir,
        "config": Consts.ConfigDir,
        "self": os.path.dirname(json_path)
    }
    if isinstance(args, str):
        return args.format(**format_map)
    elif isinstance(args, dict):
        return {
            key: replace_placeholder_with_real_value(json_path, value)
            for key, value in args.items()
        }
    elif isinstance(args, (list, set, tuple)):
        return args.__class__(
            replace_placeholder_with_real_value(json_path, v) for v in args)
    else:
        return args


class ConfigBase(object):
    def Update(self, **args):
        for key, value in self.__dict__.items():
            if key in args:
                new = args[key]
                if isinstance(value, ConfigBase):
                    setattr(self, key, value.Update(**new))
                elif isinstance(value, dict):
                    if key in _config_map:
                        setattr(
                            self, key, {
                                k: _config_map[key][0]().Update(**v)
                                for k, v in new.items()
                            })
                    else:
                        setattr(self, key, new)
                elif isinstance(value, (list, set, tuple)):
                    if key in _config_map:
                        setattr(
                            self, key,
                            value.__class__(_config_map[key][0]().Update(**v)
                                            for v in new))
                    else:
                        setattr(self, key, value.__class__(v for v in new))
                elif key in _config_map:
                    hint_type = _config_map[key]
                    if isinstance(hint_type, dict):
                        setattr(self, key, {
                            k: hint_type[0]().Update(**v)
                            for k, v in new.items()
                        })
                    elif isinstance(hint_type, (list, set, tuple)):
                        setattr(
                            self, key,
                            value.__class__(hint_type[0]().Update(**v)
                                            for v in new))
                    elif isinstance(hint_type, type(ConfigBase)):
                        setattr(self, key, hint_type().Update(**new))
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

class OptimizerConfig(ConfigBase):
    @property
    def Type(self) -> str:
        return self.type

    @property
    def LearningRate(self) -> Union[float, Dict[str, Union[int, float, str]]]:
        return self.lr

    @property
    def ClipGradients(self) -> Union[float, Dict[str, dict]]:
        return None

    @property
    def Appliers(self) -> Dict[str, dict]:
        return None

    @property
    def Args(self) -> Dict[str, object]:
        return self.args

    def __init__(self):
        self.type: str = None
        self.lr: float = 1.0
        self.args: Dict[str, object] = dict()


class HParamsConfig(ConfigBase):

    def __init__(self):
        self.m: int = 4
        self.k: int = 256
        self.searchStep: int = 1
        self.numSamples: int = 1

    def Update(self, **args):
        super().Update(**args)
        assert self.m > 0
        assert self.searchStep > 0
        assert self.numSamples > 0
        assert (self.k & (self.k - 1) == 0) and self.k != 0
        return self

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
    def SearchStep(self) -> int:
        """Search step.

        Returns:
            int -- Search steps
        """
        return self.searchStep

    @property
    def NumSamples(self) -> int:
        """Number of drawed samples.

        Returns:
            int -- num samples
        """
        return self.numSamples


class Config(ConfigBase):
    def __init__(self):
        self.hParams: HParamsConfig = HParamsConfig()
        self.optimizers: Dict[str, OptimizerConfig] = {
            "transformer": OptimizerConfig(),
            "codebook": OptimizerConfig()
        }
        self.epoch: int = 10
        self.r: int = -1
        self.vRamFraction: int = 300
        self.dataset: str = None
        self.port: int = 12345
        self.virtualSplit: bool = False
        self.gpus: int = 1
        self.wantsMore: bool = False
        self.useVRam: int = -1

    @staticmethod
    def Read(config_path):
        with open(config_path, "r") as fp:
            args = json.load(fp)
        args = replace_placeholder_with_real_value(config_path, args)
        return Config().Update(**args)

    def Update(self, **args):
        super().Update(**args)
        assert self.port > 0 and self.port < 65536
        assert self.epoch > 0
        return self

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
    def R(self) -> int:
        return self.r

    @property
    def Dataset(self) -> str:
        return self.dataset

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def VRamFraction(self) -> int:
        return self.vRamFraction

    @property
    def VirtualSplit(self) -> bool:
        return self.virtualSplit

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

    @property
    def UseVRam(self) -> int:
        return self.useVRam


_config_map: Dict[str, Type] = {
    "optimizers": {
        0: OptimizerConfig
    },
    "hParams": HParamsConfig
}

_optimMap: Dict[str, Type[tf.keras.optimizers.Optimizer]] = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "gaAdam": GradientAccumulatedAdam
}

_lrMap: Dict[str, Type[tf.keras.optimizers.schedules.LearningRateSchedule]] = {
    "exponentialDecay": tf.keras.optimizers.schedules.ExponentialDecay
}


def OptimizersFromDict(optimDict: Dict[str, OptimizerConfig], distributedBoost: float):
    return { key: _optimizerFromConfig(value, distributedBoost) for key, value in optimDict.items() }

def _optimizerFromConfig(optimConfig: OptimizerConfig, distributedBoost: float):
    assert isinstance(optimConfig, OptimizerConfig)
    if isinstance(optimConfig.LearningRate, (int, float)):
        learningRate = float(optimConfig.LearningRate)
    else:
        lrType = _lrMap[optimConfig.LearningRate.pop("type")]
        learningRate = lrType(**optimConfig.LearningRate)
        if hasattr(learningRate, "initial_learning_rate"):
            setattr(learningRate, "initial_learning_rate", getattr(learningRate, "initial_learning_rate") * distributedBoost)

    return _optimMap[optimConfig.Type](learning_rate=learningRate, **optimConfig.Args)
