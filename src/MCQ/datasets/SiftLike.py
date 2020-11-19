import os
from logging import Logger
import functools

import numpy as np
import torch
from torch.utils.data import Dataset

from MCQ import Consts
from MCQ.datasets.utils import fvecs_read, ivecs_read


class SiftLike(Dataset):
    def __init__(self, dataset: str = "SIFT/1M", logger: Logger = None, fileType="npy"):
        super().__init__()
        self._path = os.path.join(Consts.DataDir, dataset)
        self._logger = logger or Consts.Logger
        self.data = None
        fvecsFn = lambda x: fvecs_read(os.path.join(self._path, f"{x}.fvecs"))
        ivecsFn = lambda x: fvecs_read(os.path.join(self._path, f"{x}.ivecs"))
        npLoadFn = lambda x: np.load(os.path.join(self._path, f"{x}.npy"))
        self._readFn = {
            "train": fvecsFn if fileType == "vecs" else npLoadFn,
            "encode": fvecsFn if fileType == "vecs" else npLoadFn,
            "query": fvecsFn if fileType == "vecs" else npLoadFn,
            "gt": ivecsFn if fileType == "vecs" else npLoadFn
        }

    def state_dict(self) -> dict:
        return {
            "data": self.data
        }

    def load_state_dict(self, stateDict: dict) -> None:
        try:
            device = self.data.device
            del self.data
        except AttributeError:
            device = None
        self.data = stateDict["data"]
        self._logger.debug("Load train data: %s", self.data)
        if device is not None:
            self.data = self.data.to(device)

    def Train(self, length: int = -1, device: str = "cuda"):
        try:
            del self.data
        except AttributeError:
            pass
        data = self._readFn["train"]("train")
        np.random.shuffle(data)
        if length > 0:
            data = data[:length]
        self.data = torch.flatten(torch.from_numpy(data), start_dim=1).float().to(device)
        return self

    def Encode(self, device: str = "cpu"):
        try:
            del self.data
        except AttributeError:
            pass
        self.data = torch.flatten(torch.from_numpy(self._readFn["encode"]("base")), start_dim=1).float().to(device)
        return self

    def Query(self, device: str = "cpu"):
        try:
            del self.data
        except AttributeError:
            pass
        self.data = torch.flatten(torch.from_numpy(self._readFn["query"]("query")), start_dim=1).float().to(device)
        return self

    def Gt(self, device: str = "cpu"):
        try:
            del self.data
        except AttributeError:
            pass
        self.data = torch.from_numpy(self._readFn["gt"]("gt")).to(device)
        if len(self.data.shape) == 1:
            self.data = self.data[:, None]
        if self.data.shape[-1] > 1:
            self.data = self.data[:, :1]
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape
