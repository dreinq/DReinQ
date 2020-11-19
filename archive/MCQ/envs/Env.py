import tensorflow as tf
import numpy as np

from MCQ import Consts
from MCQ.layers import Codebook
from MCQ.metrics import QuantizationError
from MCQ.utils.lsqr import Solver


class Env(object):
    def __init__(self, distributedStrategy, **kwargs):
        self._codebook = Codebook(kwargs["M"], kwargs["K"], kwargs["D"], kwargs["initialCodebook"])
        self._distributedStrategy = distributedStrategy
        self._trainDatas = None
        self._oldQError = None
        self._patched = False
        self._mean = tf.keras.metrics.Mean(name="QuantizationError")
        self._solver = Solver(kwargs["M"], kwargs["K"])

    @property
    def Codebook(self):
        return self._codebook.Raw

    def Reset(self):
        self._minReward = 1e20
        self._notImprove = 0

    def Step(self, trainDatas, assignCodes):
        if self._trainDatas is None:
            self._trainDatas = trainDatas
        else:
            assert np.mean(np.sum((self._trainDatas - trainDatas) ** 2 ,-1)) < 1e-12, "Order of sampled data does not guarantee"
        # dataset = tf.data.Dataset.from_tensor_slices((trainDatas, assignCodes)).repeat(2).shuffle(5000).batch(Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        # dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)
        # self.HotPatch(dataset.element_spec)
        # for trainData, assignCode in dataset:
        #     self._updateCodebook(trainData, assignCode)
        # newCodebook = self._codebook.Raw.numpy()
        newCodebook = self._solver.solve(self._trainDatas, assignCodes)
        self._codebook.set_weights([newCodebook])
        newQError = QuantizationError(self._trainDatas, newCodebook, assignCodes).astype(np.float32)
        if self._oldQError is None:
            self._oldQError = newQError
            self._meanQE = np.mean(self._oldQError)
        # delta = (self._oldQError - newQError)
        # delta[delta > 0] /= np.max(delta, -1)
        # delta[delta < 0] /= -np.min(delta, -1)
        rewards = 2. * (self._oldQError - newQError) / self._meanQE
        # delta *= 5.0
        # rewards = ((self._oldQError - newQError) > 0).astype(np.float32)
        # rewards = np.tanh(delta)
        # rewards[delta < 0] *= -1.0
        # rewards = (-2.0 * ((self._oldQError - newQError) < 0) + 1.0).astype(np.float32)
        print(f"Quantization error: {np.mean(newQError)}")
        # self._oldQError = newQError
        # self._oldQError -= (1 - 0.9) * (self._oldQError - newQError) # np.minimum(self._oldQError, newQError)
        # print(np.sum(rewards > 0))
        # del dataset
        # obs, reward, done
        return rewards

    def HotPatch(self, elementSpec):
        if not self._patched:
            self._updateCodebook = tf.function(input_signature=elementSpec)(self._updateCodebook)

    def _updateCodebook(self, x, b):
        def _trainStep(x, b):
            with tf.GradientTape() as t:
                quantizedFeature = self._codebook(b)
                quantizationError = tf.reduce_sum((quantizedFeature - x) ** 2, -1)
                loss = tf.nn.compute_average_loss(quantizationError, global_batch_size=Consts.GlobalBatchSize)
            grads = t.gradient(loss, self._codebook.trainable_weights)
            self._optimizer.apply_gradients(zip(grads, self._codebook.trainable_weights))
            return loss

        perReplicaQE = self._distributedStrategy.run(_trainStep, args=(x, b))
        return self._distributedStrategy.reduce(tf.distribute.ReduceOp.SUM, perReplicaQE, axis=None)
