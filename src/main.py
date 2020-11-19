
import os
from logging import Logger

import torch
torch.backends.cudnn.benchmark = True

from absl import app
from absl import flags

from MCQ.datasets import SiftLike
from MCQ import Consts, Config
from MCQ.metrics import Eval
from MCQ.utils import QueryGPU, ConfigLogging, Saver
from MCQ.algorithms import SoftActorCritic, ProximalPolicy, VCPPO
from MCQ.envs import Env
from MCQ.models import ActorCritic, GumbelActorCritic, InceptAC
from MCQ.utils.runtime import PreAllocateMem

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "", "The config.json path.")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("continue", False, "Be careful to set to true. Whether to continue last training (with current config).")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")


def main(_):
    if FLAGS.eval:
        assert FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace(), f"When --eval, --path must be set, got {FLAGS.path}."
        os.makedirs(FLAGS.path, exist_ok=True)
        saveDir = FLAGS.path
        config = Config.Read(os.path.join(saveDir, Consts.DumpConfigName))
        Test(config, saveDir)
    else:
        config = Config.Read(FLAGS.config)
        if FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace():
            os.makedirs(FLAGS.path, exist_ok=True)
            saveDir = FLAGS.path
        else:
            saveDir = os.path.join(Consts.SaveDir, config.Dataset, f"{config.HParams.M}_{config.HParams.K}")
        Train(config, saveDir)

def Test(config: Config, saveDir: str, logger: Logger = None) -> None:
    avaliableGPUAndMems = QueryGPU(needGPUs=1, wantsMore=False, needVRamEachGPU=-1)
    dataset = SiftLike(config.Dataset).Train()
    _, D = dataset.shape
    paramsForEnv = {
        "M": config.HParams.M,
        "K": config.HParams.K,
        "D": D,
        "doNormalizeOnObs": config.HParams.NormalizeObs,
        "doNormalizeOnRew": config.HParams.NormalizeRew
    }
    config.HParams.__dict__.update({'d': D})
    paramsForActorCritic = config.HParams.__dict__
    methods = {
        "PPO": ActorCritic,
        "SAC": GumbelActorCritic,
        "A2C": None,
        "INC": InceptAC,
        "NOVC": ActorCritic
    }

    ConfigLogging(saveDir, Consts.LoggerName, "DEBUG" if FLAGS.debug else "INFO", rotateLogs=-1, logName="eval")

    (logger or Consts.Logger).info(str(config))
    runner = Eval(False, os.path.join(saveDir, Consts.CheckpointName), dataset, Env(**paramsForEnv), methods[config.Method](**paramsForActorCritic))
    runner.Test()

def Train(config: Config, saveDir: str, logger: Logger = None) -> None:
    avaliableGPUAndMems = QueryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.UseVRam + 256) if config.UseVRam > 0 else -1)

    dataset = SiftLike(config.Dataset).Train()
    _, D = dataset.shape

    paramsForEnv = {
        "M": config.HParams.M,
        "K": config.HParams.K,
        "D": D,
        "doNormalizeOnObs": config.HParams.NormalizeObs,
        "doNormalizeOnRew": config.HParams.NormalizeRew
    }

    config.HParams.__dict__.update({"d": D})

    paramsForActorCritic = config.HParams.ParamsForActorCritic
    paramsForActorCritic.update({'d': D})

    saver = Saver(config, saveDir, reserve=FLAGS.get_flag_value("continue", False))

    ConfigLogging(saver.SaveDir, Consts.LoggerName, "DEBUG" if FLAGS.debug else "INFO", rotateLogs=-1)

    methods = {
        "PPO": (VCPPO, ActorCritic),
        "SAC": (SoftActorCritic, GumbelActorCritic),
        "INC": (ProximalPolicy, InceptAC),
        "A2C": None,
        "NOVC": (ProximalPolicy, ActorCritic)
    }

    (logger or Consts.Logger).info(str(config))
    reinforce = methods[config.Method][0](Env(summaryWriter=saver, **paramsForEnv), methods[config.Method][1](**paramsForActorCritic), len(avaliableGPUAndMems), config.HParams, saver)

    optimizerAndSchedulerFns = config.OptimAndSchFns
    reinforce.Run(dataset, optimizerAndSchedulerFns, FLAGS.get_flag_value("continue", False), config.EvalStep)


if __name__ == "__main__":
    app.run(main)
