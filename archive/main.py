
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "", "The config.json path")
flags.DEFINE_boolean("continue", False, "Be careful to set true. Whether to continue last training (with current config)")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")

def main(_):

    import os
    import datetime
    from typing import List, Type
    import shutil
    import math

    import tensorflow as tf
    tf.debugging.enable_check_numerics()
    import numpy as np

    from MCQ.datasets import Sift
    from MCQ import Consts, Config, OptimizerConfig, OptimizersFromDict
    from MCQ.utils import QueryGPU, SplitToVirtualGPUs, RotateItems, SaveAndRestore, HotPatch
    from MCQ.models import Estimator
    from MCQ.layers import Codebook
    from MCQ.criterions import QuantizationError
    from MCQ import PPO, Reinforce
    from MCQ.envs import Env
    from MCQ.optimizers import GradientAccumulation, GradientAccumulatedAdam
    from MCQ.metrics import QuantizationError

    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    assert os.path.exists(FLAGS.config), "Given config is not exist."

    config = Config.Read(FLAGS.config)

    avaliableGPUAndMems = QueryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=((config.UseVRam + 256) if config.UseVRam > 0 else -1))
    minMem = min(avaliableGPUAndMems, key=lambda item:item[1])[1]
    batchSize = minMem // config.VRamFraction
    vRamTotal = batchSize * config.VRamFraction

    if vRamTotal > 0 and config.UseVRam > 0 and config.VirtualSplit:
        SplitToVirtualGPUs(avaliableGPUAndMems, vRamTotal)

    mirroredStrategy = tf.distribute.MirroredStrategy() if len(avaliableGPUAndMems) > 1 else tf.distribute.OneDeviceStrategy(device="/gpu:0")

    sift = Sift()
    dataset = sift(mode=Consts.Mode.Train)[:1000]
    N, D = dataset.shape

    Consts.GlobalBatchSize = (batchSize * mirroredStrategy.num_replicas_in_sync)
    # align codebook assignment
    for i in range(Consts.GlobalBatchSize, 0, -1):
        if N % i == 0:
            Consts.GlobalBatchSize = i
            break

    print(Consts.GlobalBatchSize)

    def _replaceValue(feedDict):
        for key, value in feedDict.items():
            if isinstance(value, dict):
                feedDict[key] = _replaceValue(value)
            elif isinstance(value, OptimizerConfig):
                value.args = _replaceValue(value.Args)
                if isinstance(value.LearningRate, dict):
                    value.lr = _replaceValue(value.LearningRate)
            if value == "EPOCH_STEPS":
                feedDict[key] = dataset.shape[0] // Consts.GlobalBatchSize * config.HParams.SearchStep
        return feedDict

    config.optimizers = _replaceValue(config.Optimizers)

    with mirroredStrategy.scope():
        optimizers = OptimizersFromDict(config.Optimizers, math.sqrt(mirroredStrategy.num_replicas_in_sync))

        try:
            shutil.rmtree(os.path.join(Consts.SaveDir, "summary"))
        except:
            pass

        summaryWriter = tf.summary.create_file_writer(os.path.join(Consts.SaveDir, "summary"))

        env = Env(distributedStrategy=mirroredStrategy, M=config.HParams.M, K=config.HParams.K, D=D, initialCodebook=None)

        reinforce = PPO(optimizers, env, mirroredStrategy, Consts.GlobalBatchSize, summaryWriter, M=config.HParams.M, K=config.HParams.K, D=D, initialCodebook=None)

        savePath = os.path.join(Consts.SaveDir, "ckpt")
        _, manager = SaveAndRestore(savePath, FLAGS.get_flag_value("continue", False), optimizer=optimizers["network"], actor=reinforce._actor, critic=reinforce._critic)

        reinforce.Run(dataset, manager)

if __name__ == "__main__":
    app.run(main)