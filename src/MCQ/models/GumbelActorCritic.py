import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from MCQ.layers import GumbelSoftmax

from MCQ.models import PolicyTransformer, ValueTransformer


class SharedNet(nn.Module):
    def __init__(self, m, k, d, hiddenDim, rate=0.1):
        super().__init__()
        self._m = m
        self._k = k
        self._d = d
        self._xDense = nn.Linear(d, hiddenDim)
        self._xNorm = nn.LayerNorm(hiddenDim, 1e-6)
        self._xDropout = nn.Dropout(rate)
        self._cDense = nn.Linear(d, hiddenDim)
        self._cNorm = nn.LayerNorm(hiddenDim, 1e-6)
        self._cDropout = nn.Dropout(rate)
        self._dense = nn.Linear(k, k)
        self._norm = nn.LayerNorm(k, 1e-6)
        self._dropout = nn.Dropout(rate)

    def forward(self, x, codebook):
        # [M * K, D]
        codebook = codebook.reshape(self._m * self._k, -1)
        # [N, *]
        x = self._xDropout(self._xNorm(F.relu(self._xDense(x))))
        # [M * K, *]
        codebook = self._cDropout(self._cNorm(F.relu(self._cDense(codebook))))
        # [N, M, K]
        mixed = (x @ codebook.t()).reshape(x.shape[0], self._m, -1)
        catted = []
        for i in range(self._m):
            catted.append(self._dropout(self._norm(self._dense(mixed[:, i, ...]))))
        # M * [N, K]
        return catted

class PolicyNet(nn.Module):
    def __init__(self, m, k, d, rate=0.1):
        super().__init__()
        self._m = m
        self._k = k
        self._sqrtK = math.sqrt(k)
        self._d = d
        self._dense1 = nn.Linear(d, 2 * d)
        self._norm1 = nn.LayerNorm(2 * d, 1e-6)
        self._dropout1 = nn.Dropout(rate)
        self._dense2 = nn.Linear(2 * d, m * d)
        self._dropout2 = nn.Dropout(rate)
        self._mDense = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(m)])
        self._mDropout = nn.ModuleList([nn.Dropout(rate) for _ in range(m)])
        self._mReduce = nn.ModuleList([nn.Linear(2 * d, k) for _ in range(m)])
        self._mFinal = nn.ModuleList([nn.Linear(k, k) for _ in range(m)])
        self._gumbel = GumbelSoftmax(1.0)

    def forward(self, x, logits, greedy, soft):
        hidden = self._dropout1(self._norm1(F.relu(self._dense1(x))))
        x = self._dropout2(F.relu(self._dense2(hidden)))

        splits = torch.split(x, self._d, -1)
        samples = list()
        logProbs = list()
        for i in range(self._m):
            sp = splits[i]
            sp = self._mDropout[i](F.relu(self._mDense[i](sp)))
            sp = F.relu(self._mReduce[i](sp + hidden))
            mLogits = self._mFinal[i](sp)
            if logits is None:
                if greedy:
                    sample = torch.nn.functional.one_hot(torch.argmax(mLogits, -1), self._k).float()
                else:
                    # [N, ]
                    sample = self._gumbel(mLogits, None, soft)
                # [N, ]
                logProb = self._gumbel(mLogits, sample, soft)
                samples.append(sample)
            else:
                logProb = self._gumbel(mLogits, logits[:, i * self._k:(i + 1) * self._k], soft)
            logProbs.append(logProb)
        logits = torch.cat(samples, -1) if logits is None else logits
        logProbs = sum(logProbs)
        return logits, logProbs


class ValueNet(nn.Module):
    def __init__(self, m, k, d, rate = 0.1):
        super().__init__()
        self._m = m
        self._k = k
        self._d = d
        self._shared = SharedNet(m, k, d, d)
        self._dense1 = nn.Linear(d, 1)
        self._dense2 = nn.Linear(d, 1)
        self._xDense = nn.Linear(d, 1)
        self._cDense = nn.Linear(d, 1)
        self._mReduce = nn.ModuleList([nn.Linear(d, 1) for _ in range(m)])
        self._final = nn.ModuleList([nn.Linear(k, k) for _ in range(m)])
        self._cDense1 = nn.ModuleList([nn.Linear(d, k) for _ in range(m)])
        self._cLN1 = nn.ModuleList([nn.LayerNorm(k, 1e-6) for _ in range(m)])
        self._cDropout1 = nn.ModuleList([nn.Dropout(rate) for _ in range(m)])
        self._cDense2 = nn.Linear(self._m, 1)
        self._cDropout2 = nn.Dropout(rate)
        self._final1 = nn.Linear(k, k)
        self._dropout = nn.Dropout(rate)
        self._reduce = nn.Linear(k, 1)

    def forward(self, x, codebook, logits):
        # M * [N, K]
        transformed = self._shared(x, codebook)
        codes = list()
        # M * [N, K]
        splits = torch.split(logits, self._k, -1)
        for i in range(self._m):
            # [N, D]
            selectCodewords = splits[i] @ codebook[i]
            # [N, K]
            c = self._cDropout1[i](self._cLN1[i](F.relu(self._cDense1[i](selectCodewords))))
            c = c + transformed[i]
            codes.append(c)
        # [N, K, M]
        codes = torch.stack(codes, -1)
        # [N, K]
        codes = self._cDropout2(self._cDense2(codes)[..., 0])
        final = self._dropout(F.relu(self._final1(codes)))
        return self._reduce(final)[..., 0]

class GumbelActorCritic(nn.Module):
    def __init__(self, numLayers, numHeads, hiddenDim, m, k, d, **ignoredArgs):
        super().__init__()
        self._policy = PolicyNet(m, k, d)
        self._value1 = ValueNet(m, k, d)
        self._value2 = ValueNet(m, k, d)

    def forward(self, x, codebook, greedy=False, logits=None, soft=False):
        logProbs = None
        if logits is None:
            # [batch, M, K], [batch, M], [batch, ]
            logits, logProbs = self._policy(x, logits, greedy, False)
        values1 = self._value1(x, codebook, logits)
        values2 = self._value2(x, codebook, logits)
        return logits, logProbs, values1, values2
