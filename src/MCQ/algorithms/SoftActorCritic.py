# pylint: disable=invalid-name,line-too-long
import itertools
import random
from math import log2

import torch
from torch import nn
from torch.utils.data import DataLoader

from MCQ.datasets import PPODataset, ReplayBuffer
from MCQ.utils import Saver, SearchNewBatchSize, SetNewLr
from MCQ.envs import Env


class SoftActorCritic:
    def __init__(self, env:Env, model, nParallels, hParams, saver):
        super().__init__()
        self._nParallels = nParallels
        self._nSamples = 16
        self._realBatchSize = self._batchSize = hParams.BatchSize
        self._codebookSize = [hParams.M, hParams.K, hParams.d]
        # clip range
        self._eps = hParams.Eps # 0.2
        self._logAlpha = torch.zeros(1, requires_grad=True, device="cuda")
        self._alpha = self._logAlpha.detach().exp()
        self._targetEntropy = -(hParams.M * log2(hParams.K))
        self._gamma = hParams.Gamma # 0.99
        self._lambda = hParams.Lambda # 0.95
        self._gradNorm = hParams.GradNorm # 0.5
        self._model = nn.DataParallel(model).cuda()
        self._shift = torch.arange(0, hParams.M * hParams.K, hParams.K).cuda()
        self._saver = saver
        self._env = env
        self._step1 = 0
        self._step2 = 0

    def Run(self, x, optimizerFn, continueTraining):
        replayBuffer = ReplayBuffer(self._batchSize, int(self._nSamples))
        policyOptimizer = optimizerFn(self._model.module._policy.parameters())
        valueOptimizer = optimizerFn(itertools.chain(self._model.module._value1.parameters(), self._model.module._value2.parameters()))
        alphaOptimizer = optimizerFn([self._logAlpha])
        alphaScheduler = torch.optim.lr_scheduler.ExponentialLR(policyOptimizer, 0.999) # torch.optim.lr_scheduler.CyclicLR(policyOptimizer, lr * 0.01, lr * 10.0, step_size_up=1400, step_size_down=1400)
        policyScheduler = torch.optim.lr_scheduler.ExponentialLR(policyOptimizer, 0.999) # torch.optim.lr_scheduler.CyclicLR(policyOptimizer, lr * 0.01, lr * 10.0, step_size_up=1400, step_size_down=1400)
        valueScheduler = torch.optim.lr_scheduler.ExponentialLR(valueOptimizer, 0.999) # torch.optim.lr_scheduler.CyclicLR(policyOptimizer, lr * 0.01, lr * 10.0, step_size_up=1400, step_size_down=1400)
        if continueTraining:
            Saver.Load(self._saver.SavePath, model=self._model, policyOptimizer=policyOptimizer, valueOptimizer=valueOptimizer, policyScheduler=policyScheduler, valueScheduler=valueScheduler)
        self._model.train()
        if self._env.DoNormalizationOnObs:
            self._mean = x.data.mean(0)
            self._std = x.data.std(0)
        dataloader = DataLoader(x, batch_size=27 * self._batchSize, shuffle=False, num_workers=0)
        i = 0
        while True:
            with torch.no_grad():
                # one more sample to erase the firt sample
                if False:
                    # self._sample(None, self._env, dataloader)
                    self._randomSample(replayBuffer, self._env, dataloader)
                else:
                    self._sample(replayBuffer, self._env, dataloader)
            i += 1
            if i > (self._nSamples // 4): # and i % (self._nSamples // 16) == 0:
                self._train(replayBuffer, policyOptimizer, valueOptimizer, alphaOptimizer)
                policyScheduler.step()
                valueScheduler.step()
                alphaScheduler.step()
                self._saver.Save(self._model, policyOptimizer=policyOptimizer, valueOptimizer=valueOptimizer, policyScheduler=policyScheduler, valueScheduler=valueScheduler)

            if i % 200 == 0:
                print("Eval")
                with torch.no_grad():
                    self._eval(self._env, dataloader)

    def _eval(self, env, dataloader):
        self._model.eval()
        codebook = env.Codebook
        B = list()
        X = list()
        for x in dataloader:
            X.append(x)
            if self._env.DoNormalizationOnObs:
                x = (x - self._mean) / self._std
            batchB, _, _, _ = self._model(x, codebook.repeat(self._nParallels, 1, 1), greedy=True)
            B.append(batchB.float())
        B = torch.cat(B, 0)
        X = torch.cat(X, 0)
        env.Eval(X, B)
        del B, X
        self._model.train()

    def _randomSample(self, replayBuffer, env, dataLoader):
        codebook = env.Codebook.cuda()
        X = list()
        # keep data order
        for x in dataLoader:
            X.append(x.detach())
        X = torch.cat(X, 0)
        B = torch.randint(0, self._codebookSize[1], (X.shape[0], self._codebookSize[0])).cuda()
        # [N, M]
        iy = B + self._shift
        # [N, 1] -> [N, M]
        ix = torch.arange(X.shape[0])[:, None].expand_as(iy).cuda()
        b = torch.zeros(B.shape[0], self._codebookSize[0] * self._codebookSize[1], device="cuda")
        b[[ix, iy]] = 1
        B = b
        rewards, _ = env.Step(X.cuda().detach(), B.cuda().detach())
        replayBuffer.Append(X.detach().cpu(), codebook.detach().cpu().clone(), B.detach().cpu(), rewards.detach().cpu())
        del codebook

    def _sample(self, replayBuffer, env, dataLoader):
        codebook = env.Codebook.cuda()
        X = list()
        B = list()
        for x in dataLoader:
            X.append(x.detach())
            x = x.cuda()
            if self._env.DoNormalizationOnObs:
                x = (x - self._mean) / self._std
                # codebook = (codebook - self._mean) / self._std
            # repeat the codebook, so it can correctly split to each parallel
            batchB, _, _, _ = self._model(x, codebook.repeat(self._nParallels, 1, 1), logits=None)
            B.append(batchB.detach())
        X = torch.cat(X, 0)
        B = torch.cat(B, 0)
        rewards, _ = env.Step(X.detach(), B.detach())
        if replayBuffer is not None:
            replayBuffer.Append(X.detach().cpu(), codebook.detach().cpu().clone(), B.detach().cpu(), rewards.detach().cpu())
        del codebook

    def _train(self, replayBuffer, policyOptimizer, valueOptimizer, alphaOptimizer):
        permute = torch.randperm(100000)
        # for i in range(self._nSamples):
        # sp = permute[i]

        batches = len(permute) // self._batchSize
        for i in range(batches):
            i = i % batches
            idx = permute[i * self._batchSize:(i + 1) * self._batchSize]
            j = 0 # random.choice(range(len(replayBuffer)))
            obs, codebook, action, reward = replayBuffer[j, idx]
            obs = obs.cuda()
            codebook = codebook.cuda()
            action = action.cuda()
            reward = reward.cuda()

            self._model.zero_grad()
            _, _, value1, value2 = self._model(obs, codebook.repeat(self._nParallels, 1, 1), logits=action)
            backup = reward + 0.0 # gamma * (1 - d) * (min(value1New, value2New) - alpha * actionNew)
            criticLoss = ((value1 - backup) ** 2).mean() + ((value2 - backup) ** 2).mean()
            criticLoss.backward()
            valueOptimizer.step()

            self._model.zero_grad()
            _, logProbs, reValue1, reValue2 = self._model(obs, codebook.repeat(self._nParallels, 1, 1), soft=True)
            actorLoss = (self._alpha * logProbs - torch.min(reValue1, reValue2)).mean()
            actorLoss.backward()
            policyOptimizer.step()

            alphaOptimizer.zero_grad()
            alphaLoss = -(self._logAlpha * (logProbs.detach() + self._targetEntropy)).mean()
            alphaLoss.backward()
            alphaOptimizer.step()
            self._alpha = self._logAlpha.detach().exp()

            if self._step2 % 10 == 0:
                self._saver.add_scalar("loss/actorLoss", actorLoss, global_step=self._step2)
                self._saver.add_scalar("loss/criticLoss", criticLoss, global_step=self._step2)
            self._step2 += 1
