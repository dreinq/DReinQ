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

class Reinforce():
    def __init__(self, optimizers: Dict[str, tf.keras.optimizers.Optimizer], env, distributedStrategy: tf.distribute.Strategy, globalBatchSize: int, summaryWriter, **kwargs):
        super().__init__()
        self._distributedStrategy = distributedStrategy
        self._globalBatchSize = globalBatchSize
        self.eps = 0.2
        self.alpha = 0.1
        self._gamma = np.float32(0.95)
        self._epoch = 0

        self._patched = dict()

        self._actor = PolicyEstimator(**kwargs)
        self._distribution = MultiCategorical()
        self._critic = ValueEstimator(**kwargs)
        self._env = env

        self._mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self._quantizationError = tf.keras.metrics.Mean('QuantizationError', dtype=tf.float32)

        self._summaryWriter = summaryWriter

        self._actorOptimizer = optimizers["actor"]
        self._criticOptimizer = optimizers["critic"]

        self._tmpCodebook = tf.Variable(initial_value=self._env.Codebook, trainable=False, name="tmpCodebook")

    def Train(self, x):
        dataset = tf.data.Dataset.from_tensor_slices(x).batch(Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)
        while True:
            samples = self.PolicyForward(dataset, 8)
            accuRewards = self.ValueEstimate(samples["trainDatas"], samples["codebooks"].pop(-1), samples["trajs"].pop(-1)[0])
            trainBatches = list()
            for i, ((assignCodes, negLogProbs), rewards, codebooks) in reversed(list(enumerate(zip(samples["trajs"], samples["rewards"], samples["codebooks"])))):
                accuRewards = rewards + self._gamma * accuRewards
                value = self.ValueEstimate(samples["trainDatas"], codebooks, assignCodes)
                advantages = accuRewards - value
                trainBatches.append((codebooks, assignCodes, advantages.astype(np.float32), negLogProbs, accuRewards))
            random.shuffle(trainBatches)
            for codebooks, assignCodes, advantages, negLogProbs, accuRewards in trainBatches:
                self.TrainActor(samples["trainDatas"], codebooks, assignCodes, advantages, negLogProbs)
                self.TrainCritic(samples["trainDatas"], codebooks, assignCodes, accuRewards)
            del samples, trainBatches

    def HotPatch(self, func, elementSpec):
        if isinstance(elementSpec, (tf.TensorSpec, tensorflow.python.distribute.values.PerReplicaSpec)):
            elementSpec = (elementSpec, )
        if func.__name__ in self._patched:
            return
        setattr(self, func.__name__, tf.function(input_signature=elementSpec)(func))
        self._patched[func.__name__] = True

    def ValueEstimate(self, trainDatas, codebooks, assignCodes):
        self._tmpCodebook.assign(codebooks)
        dataset = tf.data.Dataset.from_tensor_slices((trainDatas, assignCodes)).batch(Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)

        self.HotPatch(self._fetchValues, dataset.element_spec)

        values = list()
        for trainData, assignCode in dataset:
            value = self._fetchValues(trainData, assignCode)
            if self._distributedStrategy.num_replicas_in_sync == 1:
                values.append(value.numpy())
            else:
                values.extend(v.numpy() for v in value.values)
        return np.concatenate(values, 0)

    def PolicyForward(self, dataset, trajLength):
        self.HotPatch(self._fetchSamples, dataset.element_spec)
        samples = dict()
        for _ in range(trajLength):
            if "trainDatas" not in samples:
                trainDatas = list()
            else:
                trainDatas = None
            assignCodes = list()
            negLogProbs = list()
            for batchData in dataset:
                assignCode, negLogProb = self._fetchSamples(batchData)
                if self._distributedStrategy.num_replicas_in_sync == 1:
                    if trainDatas is not None:
                        trainDatas.append(batchData.numpy())
                    assignCodes.append(assignCode.numpy().astype(np.int32))
                    negLogProbs.append(negLogProb.numpy())
                else:
                    if trainDatas is not None:
                        trainDatas.extend(x.numpy().astype(np.float32) for x in batchData.values)
                    assignCodes.extend(c.numpy().astype(np.int32) for c in assignCode.values)
                    negLogProbs.extend(p.numpy() for p in negLogProb.values)
            if trainDatas is not None:
                # [N, D]
                trainDatas = np.concatenate(trainDatas, 0)
                samples["trainDatas"] = trainDatas
            # [N, M]
            assignCodes = np.concatenate(assignCodes, 0)
            # [N, 1]
            negLogProbs = np.concatenate(negLogProbs, 0)
            if "trajs" not in samples:
                samples["trajs"] = list()
            samples["trajs"].append((assignCodes, negLogProbs))

            if "codebooks" not in samples:
                samples["codebooks"] = list()
            samples["codebooks"].append(self._env.Codebook)
            _, reward, quantizationError = self._env.Step(samples["trainDatas"], assignCodes)
            # with self._summaryWriter.as_default():
            #     tf.summary.scalar("quantizationError", quantizationError, step=self._epoch)
            del quantizationError
            self._epoch += 1
            if "rewards" not in samples:
                samples["rewards"] = list()
            samples["rewards"].append(reward)
        return samples

    def TrainCritic(self, trainDatas, codebooks, assignCodes, rewards):
        self._tmpCodebook.assign(codebooks)
        dataset = tf.data.Dataset.from_tensor_slices((trainDatas, assignCodes, rewards)).shuffle(5000).batch(Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)
        self.HotPatch(self._trainCritic, dataset.element_spec)
        for trainData, assignCode, reward in dataset:
            loss = self._trainCritic(trainData, assignCode, reward)
            # print(loss)
        del dataset

    def TrainActor(self, trainDatas, codebooks, assignCodes, advantages, negLogProbs):
        self._tmpCodebook.assign(codebooks)
        dataset = tf.data.Dataset.from_tensor_slices((trainDatas, assignCodes, advantages, negLogProbs)).shuffle(5000).batch(Consts.GlobalBatchSize).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = self._distributedStrategy.experimental_distribute_dataset(dataset)
        self.HotPatch(self._trainActor, dataset.element_spec)
        for trainData, assignCode, advantage, prob in dataset:
            loss = self._trainActor(trainData, assignCode, advantage, prob)
            # print(loss)
        del dataset

    def PermuteCodebook(self):
        codebook = self._env.Codebook
        M, K, _ = codebook.shape
        for i in range(M):
            permutation = np.random.permutation(K)
            codebook[i] = codebook[i][permutation]
        permutation = np.random.permutation(M)
        codebook = codebook[permutation]
        self.UpdateCodebook(codebook)

    def _trainCritic(self, x, index, reward):
        def _trainStep(x, index, reward):
            with tf.GradientTape() as t:
                # [batch, ]
                value = self._critic(x, self._tmpCodebook, index)
                loss = self._mse(reward[..., None], value[..., None])
                loss = tf.nn.compute_average_loss(loss, global_batch_size=Consts.GlobalBatchSize)
            grads = t.gradient(loss, self._critic.trainable_weights)
            # grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            self._criticOptimizer.apply_gradients(zip(grads, self._critic.trainable_weights))
            return loss
        loss = self._distributedStrategy.run(_trainStep, args=(x, index, reward))
        return self._distributedStrategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

    def _trainActor(self, x, index, advantage, oldNegLogProb):
        def _trainStep(x, index, advantage):
            with tf.GradientTape() as t:
                # [batch, 1]
                logits = self._actor(x, self._tmpCodebook, training=True)
                negLogProb = self._distribution.NegLogP(logits, index)
                entropy = self._distribution.Entropy(logits)
                loss = negLogProb * advantage
                # loss = tf.clip_by_value(loss, -100.0, tf.float32.max)
                loss -= self.alpha * entropy
                loss = tf.nn.compute_average_loss(loss, global_batch_size=Consts.GlobalBatchSize)
            grads = t.gradient(loss, self._actor.trainable_weights)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            self._actorOptimizer.apply_gradients(zip(grads, self._actor.trainable_weights))
            return loss
        loss = self._distributedStrategy.run(_trainStep, args=(x, index, advantage))
        return self._distributedStrategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

    def _fetchSamples(self, x):
        def _fetchStep(x):
            # [batch, M, K]
            logits = self._actor(x, self._env.Codebook, training=True)
            # [batch, M], [batch, ]
            sampledY, negLogProb = self._distribution(logits)
            return (sampledY, negLogProb)

        assignCode, negLogProb = self._distributedStrategy.run(_fetchStep, args=(x,))
        return assignCode, negLogProb

    def _fetchValues(self, x, assignCode):
        def _fetchStep(x, assignCode):
            value = self._critic(x, self._tmpCodebook, assignCode, True)
            return value
        value = self._distributedStrategy.run(_fetchStep, args=(x, assignCode))
        return value
