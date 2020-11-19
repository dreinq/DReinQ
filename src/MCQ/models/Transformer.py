import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from MCQ.layers import Threshold


class PolicyTransformer(nn.Module):
    def __init__(self, numLayers, numHeads, hiddenDim, M, K, D, rate=0.1):
        super().__init__()
        self._numCodebooks = M
        self._numCodewords = K
        encoderLayer = nn.TransformerEncoderLayer(D, numHeads, hiddenDim, rate)
        self._encoder = nn.TransformerEncoder(encoderLayer, numLayers, norm=nn.LayerNorm(D, 1e-6))
        decoderLayer = nn.TransformerDecoderLayer(D, numHeads, hiddenDim, rate)
        self._decoder = nn.TransformerDecoder(decoderLayer, numLayers, norm=nn.LayerNorm(D, 1e-6))
        self._finalLayer = nn.Linear(D, self._numCodewords)
        self._threshold = Threshold()

    def forward(self, x: torch.Tensor, codebook: torch.Tensor, assignCodes: torch.Tensor):
        # (M * K, 1, D)
        flattenCodebook = codebook.reshape(-1, 1, x.shape[-1])
        # (M * K, 1, D)
        encoderOutput = self._encoder(flattenCodebook)

        # initial sequence, [1, N, D]
        sequence = x.unsqueeze(0)

        encodedCodebook = encoderOutput.reshape(self._numCodebooks, self._numCodewords, -1)
        # [M * K, N, D]
        encoderOutput = encoderOutput.expand(encoderOutput.shape[0], x.shape[0], encoderOutput.shape[2])

        logits = list()
        samples = list()
        logProbs = list()
        entropies = list()

        for i in range(self._numCodebooks):
            # lookAheadMask = torch.triu(torch.ones([sequence.shape[0], sequence.shape[0]]).cuda())
            # pick last value of decoded output: [1, N, D]
            mDecoded = self._decoder(sequence, encoderOutput)[-1:, ...]
            # [N, K]
            mLogits = self._finalLayer(mDecoded[0])
            # mLogits /= math.sqrt(mLogits.shape[-1])
            mLogits = self._threshold(mLogits)
            logits.append(mLogits)
            distribution = Categorical(logits=mLogits)
            if assignCodes is None:
                # [N, ]
                sample = distribution.sample((1, )).t_().squeeze()
                # [N, ]
                logProb = distribution.log_prob(sample)
                samples.append(sample)
                selectedCodeword = encodedCodebook[i, sample]
            else:
                entropy = distribution.entropy()
                entropies.append(entropy)
                logProb = distribution.log_prob(assignCodes[:, i])
                selectedCodeword = encodedCodebook[i, assignCodes[:, i]]
            logProbs.append(logProb)
            sequence = torch.cat((sequence, selectedCodeword[None, ...]), 0)
        entropies = None if assignCodes is None else torch.stack(entropies, -1).sum(-1)
        assignCodes = torch.stack(samples, -1) if assignCodes is None else assignCodes
        logProbs = torch.stack(logProbs, -1)
        # M * [N, K] -> [N, M, K]
        return torch.stack(logits, 1), assignCodes, -logProbs.sum(-1), entropies


class ValueTransformer(nn.Module):
    def __init__(self, numLayers, numHeads, hiddenDim, M, K, D, rate=0.1):
        super().__init__()
        self._numCodebooks = M
        self._numCodewords = K
        encoderLayer = nn.TransformerEncoderLayer(D, numHeads, hiddenDim, rate)
        self._encoder = nn.TransformerEncoder(encoderLayer, numLayers)
        decoderLayer = nn.TransformerDecoderLayer(D, numHeads, hiddenDim, rate)
        self._decoder = nn.TransformerDecoder(decoderLayer, numLayers)
        self._final = nn.Linear((M + 1) * D, D)
        self._reduce = nn.Linear(D, 1)

    def forward(self, x: torch.Tensor, codebook: torch.Tensor, assignCodes: torch.Tensor):
        # (M * K, 1, D)
        flattenCodebook = torch.reshape(codebook, (-1, 1, x.shape[-1]))
        # (M * K, 1, D)
        encoderOutput = self._encoder(flattenCodebook)

        # encodedCodebook = encoderOutput.reshape(self._numCodebooks, self._numCodewords, -1)

        # initial sequence, [1, batch, D]
        sequence = torch.unsqueeze(x, 0)
        # [M, K, D]
        # encodedCodebook = torch.reshape(encoderOutput, codebook.shape)
        # Get embedding sequence: [batch, M+1, D]
        for i in range(self._numCodebooks):
            # [batch, D] = [K, D] gathered by [batch, ]
            selectedCodeword = codebook[i, assignCodes[:, i]]
            # [i+1, batch, D] <- [i, batch, D] + [1, batch, D]
            sequence = torch.cat((sequence, selectedCodeword[None, ...]), 0)

        # [M * K, batch, D]
        encoderOutput = encoderOutput.expand(encoderOutput.shape[0], x.shape[0], encoderOutput.shape[2])
        # lookAheadMask = torch.triu(torch.ones([sequence.shape[0], sequence.shape[0]]).cuda())
        # [batch, (M + 1) * D]
        decoded = self._decoder(sequence, encoderOutput).permute(1, 0, 2).reshape(x.shape[0], -1)
        return self._reduce(self._final(decoded))[:, 0]
