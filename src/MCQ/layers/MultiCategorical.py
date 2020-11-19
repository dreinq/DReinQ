import torch
from torch import nn
from torch.distributions import Categorical


class MultiCategorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        samples = list()
        logProbs = list()
        # M
        for i in range(logits.shape[1]):
            # logits: [N, K]
            distribution = Categorical(logits=logits[:, i])
            # [N, ]
            sample = distribution.sample((1, )).t_().squeeze()
            # [N, ]
            logProb = distribution.log_prob(sample)
            samples.append(sample)
            logProbs.append(logProb)
        # [N, M]
        samples = torch.stack(samples, -1)
        # [N, M]
        logProbs = torch.stack(logProbs, -1)
        #      [batch, M], [N, ]
        return samples, -logProbs.sum(-1)

    #      [batch, M, K], [batch, M]
    def NegLogP(self, logits, index):
        logProbs = list()
        for i in range(logits.shape[1]):
            distribution = Categorical(logits=logits[:, i])
            logProbs.append(distribution.log_prob(index[:, i]))
        # [N, M]
        logProbs = torch.stack(logProbs, -1)
        #      [batch, ]
        return -logProbs.sum(-1)

    def Entropy(self, logits):
        entropies = list()
        for i in range(logits.shape[1]):
            distribution = Categorical(logits=logits[:, i])
            entropies.append(distribution.entropy())
        # [N, M]
        entropies = torch.stack(entropies, -1)
        #      [N, ]
        return entropies.sum(-1)
