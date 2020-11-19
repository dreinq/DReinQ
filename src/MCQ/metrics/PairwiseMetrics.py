import torch

def L2DistanceWithNorm(A: torch.Tensor, B: torch.Tensor):
    diff = ((A.unsqueeze(1) - B) ** 2).sum(2)
    maxi, _ = diff.max(1, keepdim=True)
    norm = diff / maxi
    return norm

def CosineSimilarity(A: torch.Tensor, B: torch.Tensor):
    ...