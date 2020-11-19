import os

import tensorflow as tf
import numpy as np

from MCQ.abstracts import Dataset
from MCQ import Consts
from MCQ.datasets.utils import fvecs_read, ivecs_read


class Sift(Dataset):
    SCALE = ("10K", "1M", "1B")

    def __init__(self, scale: str = "1M"):
        assert scale in self.SCALE, f"Scale incorrect, expect {self.SCALE} got {scale}."
        super().__init__(os.path.join(Consts.DataDir, f"SIFT", scale))

    def __call__(self, mode):
        if mode == Consts.Mode.Train:
            data = fvecs_read(os.path.join(self._path, 'sift_learn.fvecs'))
            np.random.shuffle(data)
        elif mode == Consts.Mode.Encode:
            data = fvecs_read(os.path.join(self._path, 'sift_base.fvecs'))
        elif mode == Consts.Mode.Test:
            data = fvecs_read(os.path.join(self._path, 'sift_query.fvecs'))
        # dataset = tf.data.Dataset.from_tensor_slices(data)
        # if self._mode == Consts.Mode.Train:
        #     dataset = dataset.shuffle(len(data) // 10)
        return data  # dataset.batch(self._batch_size)
