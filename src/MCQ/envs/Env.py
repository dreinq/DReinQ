import multiprocessing as mp
from logging import Logger
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MCQ.metrics import QuantizationError
from MCQ.runtime import StreamingMeanVar, RunningMeanVar
from MCQ.solvers import SSolver, ISolver, LSolver
from MCQ.utils.logging import PPrint, WaitingBar
from MCQ import Consts
from MCQ.base import Restorable


class Env(StreamingMeanVar):
    def __init__(self, M: int, K: int, D: int, summaryWriter: SummaryWriter = None, doNormalizeOnRew: bool = False, doNormalizeOnObs: bool = False, logger: Logger = None):
        super().__init__()
        self.logger = logger or Consts.Logger
        self._step = 0
        self.summaryWriter = summaryWriter
        self._m = M
        self._k = K
        self._d = D
        self._codebook = torch.randn((M, K, D), device="cuda")
        self.solver = SSolver(M, K, backend="scipy")
        self._oldQError = None
        self._firstRun = True
        self._doNormalizeOnObs = doNormalizeOnObs
        self._doNormalizeOnRew = doNormalizeOnRew

        self._obsMean = None
        self._obsStd = None
        self._meanQE = None
        self._currentQEStat = None

    def __str__(self):
        return PPrint({
            "M": self._m,
            "K": self._k,
            "D": self._d,
            "Solver": self.solver.Summary(),
            "Normalize Obs": self._doNormalizeOnObs,
            "Normalize Rew": self._doNormalizeOnRew
            })

    def load_state_dict(self, stateDict: dict) -> None:
        super().load_state_dict(stateDict)
        self.logger.debug("Load with %s", { key: self.__dict__[key] for key in stateDict.keys() })

    @property
    def CurrentQEStat(self) -> float:
        return self._currentQEStat.item()

    @property
    def DoNormalizationOnObs(self) -> bool:
        return self._doNormalizeOnObs

    @property
    def DoNormalizationOnRew(self) -> bool:
        return self._doNormalizeOnRew

    @property
    def Codebook(self) -> torch.Tensor:
        return self._codebook

    def PutObsMeanStd(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._obsMean = mean
        self._obsStd = std
        self.logger.debug("Put obs mean and std in env\r\nmean: %s\r\nstd: %s", self._obsMean, self._obsStd)

    def _estimateQEStat(self, qError):
        if self._currentQEStat is None:
            self._currentQEStat = qError.clone()
        else:
            self._currentQEStat = torch.min(self._currentQEStat, qError)

    @torch.no_grad()
    def Step(self, x: torch.Tensor, b: torch.Tensor, logStat: bool = True) -> (torch.Tensor, torch.Tensor):
        newCodebook = self.solver.solve(x, b, alternateWhenOutlier=True)
        if b.shape[-1] == self._m * self._k:
            newQError = ((x - b @ newCodebook.reshape(self._m * self._k, -1)) ** 2).sum(-1)
        else:
            newQError = QuantizationError(x.cuda(), newCodebook, b.cuda())
        if self._oldQError is None:
            self._oldQError = newQError
            self._meanQE = self._oldQError.mean()
        if self._doNormalizeOnRew:
            rewards = (self._oldQError - newQError)
            if self._firstRun:
                self._firstRun = False
            else:
                if logStat:
                    _, variance = self.Estimate(("reward", rewards), (0, ))
                    # TODO: mean or not mean?
                    rewards = rewards / (variance + 1e-8).sqrt()
                else:
                    rewards = rewards / (self._variance["estimate/reward"] + 1e-8).sqrt()
        else:
            # [N, ]
            rewards = (self._oldQError - newQError) / self._meanQE
        currentQError = newQError.mean()
        self.logger.debug("[%4d Train]QError: %3.2f", self._step, currentQError)
        if self.summaryWriter is not None:
            self.summaryWriter.add_scalar("env/QError", currentQError, global_step=self._step)
            self.summaryWriter.add_histogram("env/Reward", rewards, global_step=self._step)
        self._step += 1
        if self._doNormalizeOnObs:
            # mean, variance = self.Estimate(("codebook", newCodebook), (1, ))
            # newCodebook = (newCodebook - mean) / (variance + 1e-8).sqrt()
            if not hasattr(self, "_obsMean"):
                raise AttributeError(f"Not feed obs mean and var with DoNormalizationOnObs = {self._doNormalizeOnObs}")
            newCodebook = (newCodebook - self._obsMean) / self._obsStd
        self._codebook.data = newCodebook
        del newCodebook
        if logStat:
            self._estimateQEStat(currentQError)
        return rewards.to(x.device), currentQError.to(x.device)

    @WaitingBar("Env eval solving...", ncols=25)
    def Eval(self, x: torch.Tensor, b: torch.Tensor, additionalMsg: str = None) -> torch.Tensor:
        newCodebook = self.solver.solve(x, b, alternateWhenOutlier=True)
        # assignCodes = self._randPerm(assignCodes)
        if b.shape[-1] == self._m * self._k:
            newQError = ((x - b @ newCodebook.reshape(self._m * self._k, -1)) ** 2).sum(-1)
        else:
            newQError = QuantizationError(x, newCodebook, b)
        self.logger.info("[%4d %s]QError: %3.2f", self._step, additionalMsg or "Eval", newQError.mean())
        if self.summaryWriter is not None:
            self.summaryWriter.add_scalar("eval/QError", newQError.mean(), global_step=self._step)
        return newCodebook, newQError
