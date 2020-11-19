import torch
from torch import nn
import math


class Threshold(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, value: torch.Tensor):
        # Similar with scaled dot-product attention
        # Use straight gradient apply
        return value.tanh() * math.log(value.shape[-1])
        # return ((value.tanh() - value).detach() + value) * math.log(value.shape[-1])
