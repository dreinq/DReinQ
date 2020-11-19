from torch.utils.data import Dataset


class PPODataset(Dataset):
    def __init__(self, batchData, batchIndex, batchB, batchNegLogProbs, batchAdv, batchValues, batchReturns):
        self._batchData = batchData
        self._batchIndex = batchIndex
        self._batchB = batchB
        self._batchNegLogProbs = batchNegLogProbs
        self._batchAdv = batchAdv
        self._batchValues = batchValues
        self._batchReturns = batchReturns

    def __len__(self):
        return len(self._batchData)

    def __getitem__(self, idx):
        return self._batchData[idx], self._batchIndex[idx], self._batchB[idx], self._batchNegLogProbs[idx], self._batchAdv[idx], self._batchValues[idx], self._batchReturns[idx]

class Zip(Dataset):
    def __init__(self, *datas):
        assert len(datas) > 0
        self.datas = datas

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datas)


class Enumerate(Dataset):
    def __init__(self, data):
        assert len(data) > 0
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]


class QEDataset(Dataset):
    def __init__(self, batchData, batchB):
        self._batchData = batchData
        self._batchB = batchB

    def __len__(self):
        return len(self._batchData)

    def __getitem__(self, idx):
        return self._batchData[idx], self._batchB[idx]
