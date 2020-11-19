from typing import List, Type, Dict
import random

import tensorflow as tf
import tensorflow.python.distribute.values
import numpy as np

from MCQ.envs import Env
from MCQ.models import PolicyEstimator, ValueEstimator
from MCQ.layers import MultiCategorical
from MCQ.criterions import QuantizationError
from MCQ import Consts
from MCQ.utils.lsqr import CInitializer
from MCQ.abstracts import HotPatch

class PPO(HotPatch):
    def __init__(self, optimizers: Dict[str, tf.keras.optimizers.Optimizer], env, distributedStrategy: tf.distribute.Strategy, globalBatchSize: int, summaryWriter, **kwargs):
        super().__init__()
        self._distributedStrategy = distributedStrategy
        self._globalBatchSize = globalBatchSize
        self._trainData = None
        # clip range
        self._eps = 0.2
        self.alpha = 0.01
        self._gamma = 0.99
        self._lambda = 0.95
        self._gradNorm = 0.5
        self._epoch = 0
        self._nSamples = 8

        self._actor = PolicyEstimator(**kwargs)
        self._distribution = MultiCategorical()
        self._critic = ValueEstimator(**kwargs)
        self._env = env

        self._mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self._quantizationError = tf.keras.metrics.Mean('QuantizationError', dtype=tf.float32)

        self._summaryWriter = summaryWriter

        self._optimizer = optimizers["network"]

        self._movingStd = None

        self._tmpCodebooks = None

    def Run(self, x, ckptManager):
        dataset = tf.data.Dataset.from_tensor_slices(x).batch(4 * Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)
        i = 0
        while True:
            samples, tmpCodebooks = self.Sample(dataset, self._nSamples)
            self._tmpCodebooks = tmpCodebooks
            with tf.device("cpu"):
                lastGaeLambda = 0.0
                # allDatasets = self._samplesToBatch(samples)
                # self.Train(allDatasets)
                nextValues = samples[-1][-2]
                batchIndex = list()
                batchB = list()
                batchNegLogProbs = list()
                batchAdv = list()
                batchValues = list()
                batchReturns = list()
                batchData = list()
                for i, B, negLogProbs, values, rewards in reversed(samples[:-1]):
                    delta = rewards + self._gamma * nextValues - values
                    advantages = lastGaeLambda = (delta + self._gamma * self._lambda * lastGaeLambda)
                    if self._movingStd is None:
                        self._movingStd = tf.math.reduce_std(advantages)
                    else:
                        self._movingStd -= (1 - 0.95) * (self._movingStd - tf.math.reduce_std(advantages))
                    normalizedAdv = advantages / (self._movingStd + 1e-10)
                    returns = advantages + values
                    nextValues = values
                    batchData.append(self._trainData)
                    batchIndex.append(tf.broadcast_to(i, tf.shape(negLogProbs)))
                    batchB.append(B)
                    batchNegLogProbs.append(negLogProbs)
                    batchAdv.append(normalizedAdv)
                    batchValues.append(values)
                    batchReturns.append(returns)

                batchData = tf.concat(batchData, 0)
                batchIndex = tf.concat(batchIndex, 0)
                batchB = tf.concat(batchB, 0)
                batchNegLogProbs = tf.concat(batchNegLogProbs, 0)
                batchAdv = tf.concat(batchAdv, 0)
                batchValues = tf.concat(batchValues, 0)
                batchReturns = tf.concat(batchReturns, 0)
            # batch, then shuffle, so it can align codebook assignment (all codebooks in a batch are same)
            trainDatas = tf.data.Dataset.from_tensor_slices((batchData, batchIndex, batchB, batchNegLogProbs, batchAdv, batchValues, batchReturns)).batch(Consts.GlobalBatchSize).shuffle(100).prefetch(tf.data.experimental.AUTOTUNE)
            trainDatas = self._distributedStrategy.experimental_distribute_dataset(trainDatas)
            self.HotPatch(self._trainNetwork, trainDatas.element_spec)
            self._train(trainDatas)
            del samples, self._tmpCodebooks, trainDatas, batchData, batchIndex, batchB, batchNegLogProbs, batchAdv, batchReturns
            ckptManager.save()
            print(f"{i} steps completed, saved.")

    @tf.function
    def _sampleFn(self, dataset):
        trainData = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        B = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        negLogProbs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        i = tf.constant(0, dtype=tf.int32)
        for batchData in dataset:
            batchB, negLogProb, value = self._fetchSamples(batchData)
            expandedTrainData = self._distributedStrategy.experimental_local_results(batchData)
            expandedB = self._distributedStrategy.experimental_local_results(batchB)
            expandednegLogProb = self._distributedStrategy.experimental_local_results(negLogProb)
            expandedvalue = self._distributedStrategy.experimental_local_results(value)
            for j in range(self._distributedStrategy.num_replicas_in_sync):
                trainData = trainData.write(i + j, expandedTrainData[j])
                B = B.write(i + j, expandedB[j])
                negLogProbs = negLogProbs.write(i + j, expandednegLogProb[j])
                values = values.write(i + j, expandedvalue[j])
            i += self._distributedStrategy.num_replicas_in_sync
        return trainData.concat(), B.concat(), negLogProbs.concat(), values.concat()

    def Sample(self, dataset, nSamples):
        self.HotPatch(self._fetchSamples, dataset.element_spec)
        samples = []
        tmpCodebooks = tf.TensorArray(tf.float32, size=nSamples, dynamic_size=False)
        for i in range(nSamples + 1):
            trainData, B, negLogProbs, values = self._sampleFn(dataset)
            if self._trainData is None:
                self._trainData = trainData
            if i < nSamples:
                tmpCodebooks = tmpCodebooks.write(i, tf.identity(self._env.Codebook))
                rewards = self._env.Step(trainData.numpy(), B.numpy())
            else:
                rewards = np.zeros((self._trainData.shape[0]), np.float32)
            samples.append((i, B, negLogProbs, values, rewards))
        #               [nSamples, M, K, D]
        return samples, tmpCodebooks.stack()

    @tf.function
    def _train(self, dataset):
        for trainData, i, batchB, oldNegLogProb, advantage, oldValue, batchReturn in dataset:
            _ = self._trainNetwork(trainData, i, batchB, oldNegLogProb, advantage, oldValue, batchReturn)

    def _trainNetwork(self, trainData, i, batchB, oldNegLogProb, advantage, oldValue, batchReturn):
        def _trainStep(trainData, i, batchB, oldNegLogProb, advantage, oldValue, batchReturn):
            tf.debugging.assert_equal(i, tf.reduce_max(i))
            with tf.GradientTape() as t:
                # [batch, 1]
                logits = self._actor(trainData, self._tmpCodebooks[i[0]], training=True)
                negLogProb = self._distribution.NegLogP(logits, batchB)
                entropy = self._distribution.Entropy(logits)
                ratio = tf.exp(oldNegLogProb - negLogProb)
                surrogate1 = -ratio * advantage
                surrogate2 = -tf.clip_by_value(ratio, 1.0 - self._eps, 1.0 + self._eps) * advantage
                lossSurrogate = tf.maximum(surrogate1, surrogate2)
                actorLoss = lossSurrogate - self.alpha * entropy
                # [batch, ]
                value = self._critic(trainData, self._tmpCodebooks[i[0]], batchB)
                valueClipped = oldValue + tf.clip_by_value(value - oldValue, -self._eps, self._eps)
                surrogate1 = self._mse(batchReturn[..., None], value[..., None])
                surrogate2 = self._mse(batchReturn[..., None], valueClipped[..., None])
                criticLoss = tf.maximum(surrogate1, surrogate2)
                loss = tf.nn.compute_average_loss(actorLoss + criticLoss, global_batch_size=Consts.GlobalBatchSize)
            grads = t.gradient(loss, self._actor.trainable_weights + self._critic.trainable_weights)
            grads = [tf.clip_by_norm(g, self._gradNorm) for g in grads]
            self._optimizer.apply_gradients(zip(grads, self._actor.trainable_weights + self._critic.trainable_weights))
            return loss
        loss = self._distributedStrategy.run(_trainStep, args=(trainData, i, batchB, oldNegLogProb, advantage, oldValue, batchReturn))
        return self._distributedStrategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

    def _fetchSamples(self, x):
        def _fetchStep(x):
            # [batch, M, K]
            logits = self._actor(x, self._env.Codebook, training=True)
            # [batch, M], [batch, ]
            assignCodes, negLogProb = self._distribution(logits)
            value = self._critic(x, self._env.Codebook, assignCodes, training=True)
            return (assignCodes, negLogProb, value)

        assignCode, negLogProb, value = self._distributedStrategy.run(_fetchStep, args=(x,))
        return assignCode, negLogProb, value
