import tensorflow as tf
import numpy as np


class Codebook(tf.keras.layers.Layer):
    def __init__(self, M: int, K: int, D: int, initialCodebook: np.ndarray):
        super().__init__()
        self._m = M
        self._k = K
        self._codebook = self.add_weight(name="codebook", shape=(self._m, self._k, D), initializer="random_normal", trainable=True)
        if initialCodebook is not None:
            self._codebook.assign(tf.convert_to_tensor(initialCodebook, dtype=tf.float32))

    def call(self, pickedIndices: tf.Tensor):
        # [M]
        ix = tf.range(self._m, dtype=pickedIndices.dtype)
        # [batch, M, 1]
        pickedIndices = pickedIndices[..., None]
        # [batch, M, 2]
        indices = tf.concat([tf.broadcast_to(ix[None, :, None], tf.shape(pickedIndices)), pickedIndices], -1)
        # [batch, M, D]
        gatheredCodewords = tf.gather_nd(self._codebook, indices)
        return tf.reduce_sum(gatheredCodewords, 1)

    def Shuffle(self, times: tf.Tensor):
        shuffledIndex = tf.TensorArray(tf.int32, size=times, dynamic_size=False)
        for i in tf.range(times):
            shuffledIndex = shuffledIndex.write(i, tf.random.shuffle(tf.range(self._m, dtype=tf.int32)))
        # [M, K, D] gatherd by [times, M]
        return tf.gather(self._codebook, shuffledIndex.stack(), axis=0)

    @property
    def Raw(self):
        return self._codebook
