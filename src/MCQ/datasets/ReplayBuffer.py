import random
from typing import Dict

import torch
from torch.utils.data import Dataset

from MCQ import Consts

class ReplayBuffer(Dataset):
    def __init__(self, batchSize, capacity):
        super().__init__()
        self._capacity = capacity
        self._batchSize = batchSize
        self._current = -1
        self._size = 0

    def Append(self, obs, codebook, act, reward):
        if self._current < 0:
            self._initial(obs, codebook, act, reward)
            self._current = -1
        self._current = (self._current + 1) % self._capacity
        self._obs[self._current] = obs
        self._act[self._current] = act
        self._codebook[self._current] = codebook
        self._reward[self._current] = reward
        self._size = min(self._size + 1, self._capacity)

    def _initial(self, obs, codebook, act, reward):
        self._obs = torch.zeros(self._capacity, *obs.shape)
        self._codebook = torch.zeros(self._capacity, *codebook.shape)
        self._act = torch.zeros(self._capacity, *act.shape)
        self._reward = torch.zeros(self._capacity, *reward.shape)

    @property
    def Size(self):
        return self._size

    @property
    def Capacity(self):
        return self._capacity

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self._obs[idx], self._codebook[idx[0]], self._act[idx], self._reward[idx]

class ReplayDict(Dataset):
    def __init__(self, capacity):
        super().__init__()
        self._capacity = capacity
        self._buffer: Dict[float, Dict[str, torch.Tensor]] = dict()
        self._sampleFrom: str = None

    def Append(self, key, **values):
        self._buffer[key] = values
        if len(self._buffer) > self._capacity:
            self._buffer.pop(max(self._buffer))

    def Check(self, eliminateFn):
        eliminateFn(self._buffer)

    @property
    def Size(self):
        return len(self._buffer)

    @property
    def Capacity(self):
        return self._capacity

    def Rollout(self):
        self._sampleFrom = random.choice([*self._buffer])
        return self._buffer[self._sampleFrom]

    def __len__(self):
        return len(self._buffer[self._sampleFrom].values()[0])

    def __getitem__(self, idx):
        contents = self._buffer[self._sampleFrom]
        return { key: value[idx] for key, value in contents.items() }
