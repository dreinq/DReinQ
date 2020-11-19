import tensorflow as tf

class QuantizationError(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, target: tf.Tensor, real: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tf.square(target - real), -1)