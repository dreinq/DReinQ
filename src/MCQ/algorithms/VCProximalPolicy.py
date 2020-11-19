
from logging import Logger
from typing import Callable, Tuple, Iterator, Dict, Union
from random import random

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from MCQ.datasets import Zip, ReplayDict
from MCQ.utils import Saver
from MCQ.utils.runtime import Timer
from MCQ.utils.logging import WaitingBar
from MCQ.envs import Env
from MCQ.Config import HParamsConfig
from MCQ import Consts
from MCQ.base import Restorable, DataParallel
from MCQ.metrics import QuantizationError

class ProximalPolicy(Restorable):
    def __init__(self, env: Env, model: nn.Module, nParallels: int, hParams: HParamsConfig, saver: Saver, logger: Logger = None):
        super().__init__()
        self.logger = logger or Consts.Logger
        # num of gpus
        self._nParallels = nParallels
        self._batchSize = hParams.BatchSize
        self._codebookSize = [hParams.M, hParams.K, hParams.d]
        # PPO clip range
        self._eps = hParams.Eps # 0.2
        # PPO entropy regularization
        self._alpha = hParams.Alpha # 0.05
        self._alphaDiscount = hParams.AlphaDiscount # 0.05
        # GAE params
        self._gamma = hParams.Gamma # 0.99
        self._lambda = hParams.Lambda # 0.95
        # policy gradient clipping
        self._gradNorm = hParams.GradNorm # 0.5
        self._icmProb = 0.01
        self._replay = 50
        self._icmScale = 1e-4

        model = model.cuda()

        # CAUTION: nn.Module.parameters() produce a generator.
        # After consume the generator once, the next time it will be empty.
        # So we call the parameter function every time we want it.
        self.policyParams = model._policy.parameters
        self.valueParams = model._value.parameters

        if nParallels > 1:
            model = DataParallel(model)
        self.model = model

        self.saver = saver
        self.env = env

        # Used when do normalization on obs
        self._obsMean = None
        self._obsStd = None
        self._obsStdRepeat = None
        self._obsMeanRepeat = None

        # other attributes for runtime
        self._step1 = 0
        self._step2 = 0
        self.ticker = Timer()
        self._icmLogs = list()

        self.logger.info("Running algorithm: PPO MOD. \r\nModel: %s\r\nEnv: %s", self.model.__class__.__name__, self.env)

        self._debugPrint("Model summary: %s" % (self.model))

    def load_state_dict(self, stateDict: dict) -> None:
        super().load_state_dict(stateDict)
        self._debugPrint("Load with: %s" % { key: self.__dict__[key] for key in stateDict.keys() })

    def _debugPrint(self, msg: str) -> None:
        self.logger.debug("[%4d] %s", self._step1, msg)

    #pylint:disable=protected-access
    def Run(self, x: Dataset, optimAndSchFns: Dict[str, Tuple[Callable[[Iterator], Optimizer], Callable[[Optimizer], torch.optim.lr_scheduler._LRScheduler]]], continueTraining: bool, evalStep: int) -> None:
        policyOptimizer = optimAndSchFns["policy"][0](self.policyParams())
        valueOptimizer = optimAndSchFns["value"][0](self.valueParams())
        self._debugPrint("Policy optimizer summary: %s" % (policyOptimizer.state_dict()))
        self._debugPrint("Value optimizer summary: %s" % (valueOptimizer.state_dict()))

        try:
            policyScheduler = optimAndSchFns["policy"][1](policyOptimizer)
            self._debugPrint("Policy scheduler summary: %s" % (policyScheduler.state_dict()))
        except TypeError:
            policyScheduler = None
            self._debugPrint("Not using policy scheduler")
        try:
            valueScheduler = optimAndSchFns["value"][1](valueOptimizer)
            self._debugPrint("Value scheduler summary: %s" % (valueScheduler.state_dict()))
        except TypeError:
            valueScheduler = None
            self._debugPrint("Not using value scheduler")

        if continueTraining:
            self.logger.info("Continue training, load checkpoint.")
            Saver.Load(self.saver.SavePath, model=self.model, env=self.env, policyOptimizer=policyOptimizer, valueOptimizer=valueOptimizer, policyScheduler=policyScheduler, valueScheduler=valueScheduler, PPO=self, dataset=x)
        self._beforeTrain(x, continueTraining)

        self.logger.info("Begin training loop")

        self._trainingLoop(x, (policyOptimizer, valueOptimizer), (policyScheduler, valueScheduler), evalStep)

    def _beforeTrain(self, x, continueTraining):
        self.model.train()
        if self.env.DoNormalizationOnObs and not continueTraining:
            self._obsMean = x.data.detach().mean(0).cuda()
            self._obsStd = x.data.detach().std(0).cuda()
            self._obsStd[self._obsStd == 0] = 1.0
            self._obsMeanRepeat = self._obsMean.repeat(self._nParallels)
            self._obsStdRepeat = self._obsStd.repeat(self._nParallels)
            self.env.PutObsMeanStd(self._obsMean, self._obsStd)

    def _trainingLoop(self, x, optims, shdrs, evalStep):
        policyOptimizer, valueOptimizer = optims
        policyScheduler, valueScheduler = shdrs
        dataLoader = DataLoader(x, batch_size=27 * self._batchSize, shuffle=False, num_workers=0)

        while True:
            # self._debugPrint("Select icm prob: {:.2%}".format(len(self._icmLogs) / self._replay))
            batchData, batchB, batchNegLogProbs, batchValues, batchRew, C = self._sample(self.env, dataLoader, False)

            self._addCodebookStats(batchB, batchValues)

            self._step1 += 1

            if self._step1 == 1:
                del batchData, batchB, batchNegLogProbs, batchRew, batchValues, C
                # skip step 1, since we calculated initial QError as baseline in step 1.
                continue

            # Do normalization on advantages with current policy (un-biased version)
            # https://arxiv.org/pdf/1811.02553.pdf Appendix A.1 "Surrogate Objectives".
            # advantage = (advantage - advantage.mean()) / advantage.std()

            # batchData, C, batchB, batchNegLogProbs, batchRew, batchValues = self._chooseReplayBufferWithProbability(batchData, C, batchB, batchNegLogProbs, batchRew, batchValues)

            trainDatas = Zip(batchData, batchB, batchNegLogProbs, batchRew, batchValues)
            trainLoader = DataLoader(trainDatas, batch_size=self._batchSize, shuffle=True, num_workers=0)
            self._train(policyOptimizer, valueOptimizer, trainLoader, C, False)
            if self._step1 % evalStep == 0:
                self._eval(self.env, dataLoader)
                # Save dataset to keep data order
                self.saver.Save(model=self.model, env=self.env, policyOptimizer=policyOptimizer, valueOptimizer=valueOptimizer, policyScheduler=policyScheduler, valueScheduler=valueScheduler, PPO=self, dataset=x)
            del batchData, batchB, batchNegLogProbs, batchRew, C, trainLoader, trainDatas

            self._shortBreak(x, policyOptimizer, valueOptimizer, policyScheduler, valueScheduler)

    def _shortBreak(self, x, policyOptimizer, valueOptimizer, policyScheduler, valueScheduler):
        try:
            policyScheduler.step()
            valueScheduler.step()
        except AttributeError:
            pass
        self._alpha *= self._alphaDiscount
        self._icmProb = min(self._icmProb * 1.001, 0.5)

        self.saver.add_scalar("stat/IterSpeed", self.ticker.Tick()[0], global_step=self._step1)

        # def _eliminateFn(bufferDict):
        #     self._debugPrint("Current QE stat: %f" % self.env._currentQEStat)
        #     for key in [*bufferDict]:
        #         if key > self.env.CurrentQEStat:
        #             bufferDict.pop(key)
        # self._icmLogs = [x for x in self._icmLogs if x < self.env.CurrentQEStat]

        # self.replayBuffer.Check(_eliminateFn)

    def _addCodebookStats(self, b, values):
        for i in range(b.shape[-1]):
            self.saver.add_histogram(f"env/Codeword{i}", b[:, i], global_step=self._step1)
        self.saver.add_histogram("stat/Value", values, global_step=self._step1)

    # def _chooseReplayBufferWithProbability(self, batchData, codebook, batchB, batchNegLogProbs, batchRew, batchValues):
    #     self._debugPrint("Replay buffer size: %d" % self.replayBuffer.Size)
    #     prob = self.replayBuffer.Size / self.replayBuffer.Capacity
    #     if self.replayBuffer.Size > 0 and random() <= prob:
    #         self._debugPrint("Select replay buffer")
    #         # replay datas with buffer value:
    #         values = self.replayBuffer.Rollout()
    #         del batchData, codebook, batchB, batchNegLogProbs, batchRew, batchValues
    #         returns = values["X"], values["C"], values["B"], values["P"], values["R"], values["V"]
    #         return tuple(torch.from_numpy(x).cuda() for x in returns)
    #     return batchData, codebook, batchB, batchNegLogProbs, batchRew, batchValues

    @torch.no_grad()
    def _eval(self, env, dataLoader):
        self._debugPrint("Eval step")
        self.model.eval()
        B = list()
        X = list()
        for x in dataLoader:
            X.append(x)
            if self.env.DoNormalizationOnObs:
                x = (x - self._obsMean) / self._obsStd
            batchB = self.model.Encode(x)
            B.append(batchB)
        B = torch.cat(B, 0)
        X = torch.cat(X, 0)
        C, qErrors = env.Eval(X, B)
        del B
        B = list()
        for x in dataLoader:
            if self.env.DoNormalizationOnObs:
                x = (x - self._obsMean) / self._obsStd
            batchB = self.model.Encode(x, icm=True, C=C.repeat(self._nParallels, 1, 1), shift=self._obsMeanRepeat, scale=self._obsStdRepeat)
            B.append(batchB)
        B = torch.cat(B, 0)
        icmQErrors = QuantizationError(X, C, B)
        self.logger.info("After ICM: %f, %.0f%% samples are better.", icmQErrors.mean(), (icmQErrors < qErrors).sum().float() / len(qErrors) * 100.)

        del B, X, C
        self.model.train()

    @torch.no_grad()
    def _sample(self, env, dataLoader, icmAssistance):
        self._debugPrint("ICM step" if icmAssistance else "Sample step")
        # repeat the C, so it can correctly split to each parallel
        C = env.Codebook.repeat(self._nParallels, 1, 1)
        X = list()
        B = list()
        negLogProbs = list()
        values = list()
        for x in dataLoader:
            X.append(x)
            if self.env.DoNormalizationOnObs:
                x = (x - self._obsMean) / self._obsStd
            if icmAssistance:
                batchB, negLogProb, value = self.model.ICM(x, C, self._obsMeanRepeat, self._obsStdRepeat, icmRange=1, greedy=False)
            else:
                batchB, negLogProb, _, value = self.model(x, C, b=None)
            B.append(batchB)
            negLogProbs.append(negLogProb)
            values.append(value)
        X = torch.cat(X, 0)
        B = torch.cat(B, 0)
        negLogProbs = torch.cat(negLogProbs, 0)
        values = torch.cat(values, 0)
        with WaitingBar(f"[{self._step1}] Env step", ncols=25):
            rewards, _ = env.Step(X, B, not icmAssistance)
        # if icmAssistance:
        #     if len(self._icmLogs) > self._replay:
        #         self._icmLogs.clear()
        #     self._icmLogs.append(qError.mean().item())
            # self.replayBuffer.Append(qError.mean().item(), X=X.detach().cpu().numpy(), C=env.Codebook.detach().cpu().numpy(), P=negLogProbs.detach().cpu().numpy(), B=B.detach().cpu().numpy(), R=rewards.detach().cpu().numpy(), V=values.detach().cpu().numpy())
        return X, B, negLogProbs, values, rewards, C

    def _train(self, policyOptimizer, valueOptimizer, dataLoader, C, icm):
        self._debugPrint("Train step")
        for trainData, batchB, oldNegLogProb, reward, oldValue in tqdm(dataLoader, leave=False, ncols=40, bar_format="{l_bar}{bar}| [" + str(self._step1) + "] Train step"):
            if self.env.DoNormalizationOnObs:
                trainData = (trainData - self._obsMean) / self._obsStd
            self.model.zero_grad()
            valueLoss = 0.5 * self._valueLoss(trainData, C, batchB, oldValue, reward)
            valueLoss.backward()
            valueOptimizer.step()

            self.model.zero_grad()
            # if False:
            #     policyLoss = self._crossEntropy(trainData, C, batchB)
            # else:
            policyLoss = self._policyLoss(trainData, C, batchB, oldNegLogProb, reward)
            # if icm:
            #     policyLoss = policyLoss * self._icmScale
            policyLoss.backward()
            if self._gradNorm > 0.0:
                nn.utils.clip_grad_norm_(self.policyParams(), self._gradNorm)
            policyOptimizer.step()

            if self._step2 % 10 == 0:
                self.saver.add_scalar("loss/PolicyLoss", policyLoss, global_step=self._step2)
                self.saver.add_scalar("loss/ValueLoss", valueLoss, global_step=self._step2)
            self._step2 += 1

    def _valueLoss(self, trainData, C, batchB, oldValue, batchReturn):
        _, _, _, value = self.model(trainData, C, b=batchB)

        valueClipped = oldValue + (value - oldValue).clamp(-self._eps, self._eps)
        surrogate1 = (batchReturn - value) ** 2
        surrogate2 = (batchReturn - valueClipped) ** 2
        valueLoss = torch.max(surrogate1, surrogate2)
        return valueLoss.mean()

    def _policyLoss(self, trainData, C, batchB, oldNegLogProb, reward):
        _, negLogProb, entropy, value = self.model(trainData, C, b=batchB)
        # Do normalization on advantages with current policy (biased version)
        # https://arxiv.org/pdf/1811.02553.pdf Appendix A.1 "Surrogate Objectives".
        with torch.no_grad():
            advantage = reward - value
            advantage -= advantage.mean()
            advantage /= advantage.std()
        ratio = torch.exp(oldNegLogProb - negLogProb)
        surrogate1 = -ratio * advantage
        surrogate2 = -ratio.clamp(1.0 - self._eps, 1.0 + self._eps) * advantage
        lossSurrogate = torch.max(surrogate1, surrogate2)
        policyLoss = lossSurrogate - self._alpha * entropy
        return policyLoss.mean()

    def _crossEntropy(self, trainData, C, batchB):
        _, negLogProb, entropy, _ = self.model(trainData, C, b=batchB)
        return self._icmScale * (negLogProb - self._alpha * entropy).mean()
