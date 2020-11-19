import tensorflow as tf

class QuantizationLogger(tf.keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs=None):
        ...