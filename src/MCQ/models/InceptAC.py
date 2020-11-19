import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from MCQ.metrics import Reconstruct, QuantizationError
from MCQ.base import DataParallel, ParallelFunction

class LayerGroup(nn.Module):
    def __init__(self, din, dout, rate):
        super().__init__()
        self._dense = nn.Linear(din, dout)
        self._norm = nn.LayerNorm(dout, 1e-6)
        self._dropout = nn.Dropout(rate)

    def forward(self, x):
        return self._dropout(self._norm(F.relu(self._dense(x))))

class LayerGroupWithoutRelu(nn.Module):
    def __init__(self, din, dout, rate):
        super().__init__()
        self._dense = nn.Linear(din, dout)
        self._norm = nn.LayerNorm(dout, 1e-6)
        self._dropout = nn.Dropout(rate)

    def forward(self, x):
        return self._dropout(self._norm(self._dense(x)))

class PyramidNet(nn.Module):
    def __init__(self, d, rate):
        super().__init__()
        self._d = d
        self._preLayer = LayerGroup(d, d, rate)
        self._left1 = LayerGroup(d, 4 * d, rate)
        self._left2 = LayerGroup(4 * d, 2 * d, rate)
        self._left3 = LayerGroup(2 * d, d, rate)
        self._transition = LayerGroup(d, d, rate)
        self._right3 = LayerGroup(d, 2 * d, rate)
        self._right2 = LayerGroup(2 * d, 4 * d, rate)
        self._right1 = LayerGroup(4 * d, d, rate)
        self._final = nn.Linear(8 * d, 8 * d)

    def forward(self, x):
        x = self._preLayer(x)
        # [N, 4 * d]
        left1 = self._left1(x)
        # [N, 2 * d]
        left2 = self._left2(left1)
        # [N, d]
        left3 = self._left3(left2)
        # [N, d]
        transition = self._transition(left3)
        # [N, 2 * d]
        right3 = self._right3(transition + x)
        # [N, 4 * d]
        right2 = self._right2(right3 + left2)
        # [N, d]
        right1 = self._right1(right2 + left1)
        # [N, 8 * d]
        catted = torch.cat([right1, right2, right3, transition], -1)
        return self._final(catted)


class InceptionBlock(nn.Module):
    def __init__(self, d, rate):
        super().__init__()
        self._d = d
        self._big1 = LayerGroup(d, 2 * d, rate)
        self._big2 = LayerGroup(2 * d, d, rate)

        self._mid1 = LayerGroup(d, d, rate)
        self._mid2 = LayerGroup(d, d, rate)

        self._sma1 = LayerGroup(d, d // 2, rate)
        self._sma2 = LayerGroup(d // 2, d // 2, rate)
        self._sma3 = LayerGroup(d // 2, d, rate)

        self._nom1 = LayerGroup(d, d, rate)

    def forward(self, x):
        big = self._big2(self._big1(x))
        mid = self._mid2(self._mid1(x))
        sma = self._sma3(self._sma2(self._sma1(x)))
        nom = self._nom1(x)
        return F.relu(x + big + mid + sma + nom)

class SharedNet(nn.Module):
    def __init__(self, m, k, d, rate):
        super().__init__()
        self._m = m
        self._k = k
        self._d = d
        self._xPyramid = InceptionBlock(d, rate)
        self._cPyramid = InceptionBlock(m * d, rate)
        self._cDense = LayerGroup(m * d, d, rate)

    def forward(self, x, codebook):
        # [N, M * D]
        codebook = codebook.reshape(-1, self._m * self._d)
        # [N, d]
        x = self._xPyramid(x)
        # [N, d]
        codebook = self._cDense(self._cPyramid(codebook))
        # [N, d]
        return x + codebook

class PolicyNet(nn.Module):
    def __init__(self, m, k, d, rate):
        super().__init__()
        self._m = m
        self._k = k
        self._sqrtK = math.sqrt(k)
        self._d = d
        self._subLayers = nn.ModuleList([nn.Sequential(InceptionBlock(d, rate), LayerGroup(d, self._m * self._k, rate), nn.Linear(self._m * self._k, self._k)) for _ in range(self._m)])

    @staticmethod
    def _scramble(a):
        """
        Return an array with the values of `a` independently shuffled along last axis
        """
        n = a.shape[-1]
        idx = torch.multinomial(torch.ones((n, ), device="cuda") / n, n)
        return a[..., idx]

    @staticmethod
    def _betterChoice(miniX, miniCodes, codebook, codebookToSearch):
        # [N, K, D]
        searchingCodebooks = codebook[codebookToSearch]
        # [N, ]
        ix = codebookToSearch
        # [N, ]
        iy = miniCodes[range(len(miniCodes)), codebookToSearch]
        # [N, D]
        codewordToReplace = codebook[[ix, iy]]
        # [N, D]
        quantized = Reconstruct(codebook, miniCodes)
        # [N, D]
        quantizeAndRemoveCodewordToReplace = quantized - codewordToReplace
        # [N, K, D]
        icmCandidates = quantizeAndRemoveCodewordToReplace[:, None, ...] + searchingCodebooks
        # [N, K] = [N, 1, D] - [N, K, D] then squared sum
        quantizationError = ((miniX[:, None, ...] - icmCandidates) ** 2).sum(-1)
        # [N, ], indices of better codeword of searched codebook
        betterSolution = quantizationError.argmin(-1)
        # overwrite new solution
        miniCodes[range(len(miniCodes)), codebookToSearch] = betterSolution

    def _miniBatchICM(self, x, b, codebook, icmRange):
        N = len(x)
        splitMiniBatchSize = N // 100
        batches = N // splitMiniBatchSize
        for i in range(batches):
            miniX = x[i * splitMiniBatchSize:(i + 1) * splitMiniBatchSize]
            miniCodes = b[i * splitMiniBatchSize:(i + 1) * splitMiniBatchSize]
            # random choose a sub-codebook to perform ICM
            # [N, ]
            # codebookToSearch = torch.randint(0, self._m, (miniX.shape[0], ))

            # _betterChoice(miniX, miniCodes, codebookToSearch)
            indices = torch.randperm(self._m, device=miniCodes.device).expand_as(miniCodes)
            for j in range(icmRange):
                self._betterChoice(miniX, miniCodes, codebook, indices[:, j])
            del indices
        remainX = x[(i + 1) * splitMiniBatchSize:]
        remainCodes = b[(i + 1) * splitMiniBatchSize:]
        if len(remainX) > 0:
            indices = torch.arange(self._m, device=remainX.device).expand_as(remainCodes)
            indices = self._scramble(indices)
            for j in range(icmRange):
                self._betterChoice(remainX, remainCodes, codebook, indices[:, j])
            del indices

    def Encode(self, x, icm, codebook, shift, scale):
        b = list()
        for i in range(self._m):
            logit = self._subLayers[i](x)
            code = torch.argmax(logit, -1)
            b.append(code)
        b = torch.stack(b, -1)
        if icm:
            assert codebook is not None
            if scale is not None:
                x = x * scale
            if shift is not None:
                x = x + shift
            self._miniBatchICM(x, b, codebook, self._m)
        return b

    def ICM(self, x, codebook, shift, scale, icmRange, greedy):
        # assert self._m > 1, f"Can't perform ICM when M = {self._m}"
        samples = list()
        logProbs = list()
        distributions = list()
        for i in range(self._m):
            logit = self._subLayers[i](x)
            distribution = Categorical(logits=logit)
            distributions.append(distribution)
            if greedy:
                sample = torch.argmax(logit, -1)
            else:
                # [N, ]
                sample = distribution.sample((1, )).t_().squeeze()
            samples.append(sample)

        b = torch.stack(samples, -1)

        if scale is not None:
            x = x * scale
        if shift is not None:
            x = x + shift

        self._miniBatchICM(x, b, codebook, icmRange)
        # compute logProbs
        for i in range(self._m):
            distribution = distributions[i]
            logProbs.append(distribution.log_prob(b[:, i]))

        negLogProbs = -sum(logProbs)
        return b, negLogProbs

    def forward(self, x, b, greedy):
        samples = list()
        logProbs = list()
        entropies = list()
        for i in range(self._m):
            logit = self._subLayers[i](x)
            distribution = Categorical(logits=logit)
            if b is None:
                if greedy:
                    sample = torch.argmax(logit, -1)
                else:
                    # [N, ]
                    sample = distribution.sample((1, )).t_().squeeze()
                # [N, ]
                logProb = distribution.log_prob(sample)
                samples.append(sample)
            else:
                entropy = distribution.entropy()
                entropies.append(entropy)
                logProb = distribution.log_prob(b[:, i])
            logProbs.append(logProb)
        entropies = None if not entropies else sum(entropies)
        b = torch.stack(samples, -1) if b is None else b
        negLogProbs = None if not logProbs else -sum(logProbs)
        return b, negLogProbs, entropies


class ValueNet(nn.Module):
    def __init__(self, m, k, d, rate):
        rate = 0.0
        super().__init__()
        self._m = m
        self._k = k
        self._d = d
        self._shared = SharedNet(m, k, d, rate)
        self._cLayers = nn.ModuleList([LayerGroup(d, d, rate) for _ in range(m)])
        self._final = LayerGroup(d, 2 * d, rate)
        self._reduce = nn.Linear(2 * d, 1)

    def forward(self, x, codebook, b):
        codes = list()
        for i in range(self._m):
            # [N, D]
            selectCodewords = codebook[i, b[:, i]]
            # [N, D]
            c = self._cLayers[i](selectCodewords)
            codes.append(c)
        # [N, M, D]
        codes = torch.stack(codes, 1)
        # [N, 8 * D]
        transformed = self._shared(x, codes)
        # [N, 2 * D]
        codes = self._final(transformed)
        return self._reduce(codes)[..., 0]

class ActorCritic(nn.Module):
    def __init__(self, m, k, d, dropoutRate=0.1, **ignoredArgs):
        super().__init__()
        self._policy = PolicyNet(m, k, d, dropoutRate)
        self._value = ValueNet(m, k, d, dropoutRate)

    @ParallelFunction
    def Encode(self, x, icm=False, codebook=None, shift=None, scale=None):
        return self._policy.Encode(x, icm, codebook, shift, scale)

    @ParallelFunction
    def ICM(self, x, codebook, shift=None, scale=None, icmRange=-1, greedy=False):
        b, negLogProbs = self._policy.ICM(x, codebook, shift, scale, self._m if icmRange < 0 else icmRange, greedy)
        values = self._value(x, codebook, b)
        return b, negLogProbs, values

    def forward(self, x, codebook, b=None, greedy=False):
        # [batch, M, K], [batch, M], [batch, ]
        b, negLogProbs, entropies = self._policy(x, b, greedy)
        values = self._value(x, codebook, b)
        return b, negLogProbs, entropies, values
