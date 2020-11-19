from typing import Union, Optional
import os
import datetime

import tensorflow as tf

from MCQ.models import PolicyTransformer, ValueTransformer
from MCQ.layers import Codebook, MultiCategorical

class PolicyEstimator(tf.keras.Model):
    def __init__(self, M, K, D, initialCodebook=None):
        super().__init__()
        self._transformer = PolicyTransformer(3, M, K, 8, 256)
        self._distribution = MultiCategorical()

    def call(self, x: tf.Tensor, codebook, temperature=1.0, training: Optional[bool] = False):
        # [batch, M, K]
        return self._transformer(x, codebook, temperature, training)

class ValueEstimator(tf.keras.Model):
    def __init__(self, M, K, D, initialCodebook=None):
        super().__init__()
        self._transformer = ValueTransformer(3, M, K, 8, 256)

    def call(self, x: tf.Tensor, codebook, assignCodes, temperature=1.0, training: Optional[bool] = False):
        return self._transformer(x, codebook, assignCodes, temperature, training)