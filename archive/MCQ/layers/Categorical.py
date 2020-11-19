import tensorflow as tf

class Categorical(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def call(self, logits):
        # [batch, ]
        samples = tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0]
        return samples, tf.nn.sparse_softmax_cross_entropy_with_logits(samples, logits)

    def NegLogP(self, logits, index):
        # [batch, ]
        return tf.nn.sparse_softmax_cross_entropy_with_logits(index, logits)

    def Entropy(self, logits):
        # logits: [batch, K]
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(tf.math.multiply_no_nan(p0, (tf.math.log(z0) - a0)), axis=-1)

class MultiCategorical(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._categorical = Categorical()

    def build(self, input_shape):
        pass

    def call(self, logits):
        samples = tf.TensorArray(tf.int32, size=tf.shape(logits)[1])
        negLogProbs = tf.TensorArray(tf.float32, size=tf.shape(logits)[1])
        for i in tf.range(tf.shape(logits)[1]):
            # [batch, ], [batch, ]
            sample, negLogProb = self._categorical(logits[:, i])
            samples = samples.write(i, sample)
            negLogProbs = negLogProbs.write(i, negLogProb)
        #      [batch, M],                    [batch, ]
        return tf.transpose(samples.stack()), tf.reduce_sum(tf.transpose(negLogProbs.stack()), -1)


    #      [batch, M, K], [batch, M]
    def NegLogP(self, logits, index):
        negLogProbs = tf.TensorArray(tf.float32, size=tf.shape(logits)[1])
        for i in tf.range(tf.shape(logits)[1]):
            # [batch, ],                             [batch, K], [batch, ]
            negLogProb = self._categorical.NegLogP(logits[:, i], index[:, i])
            negLogProbs = negLogProbs.write(i, negLogProb)
        #      [batch, ]
        return tf.reduce_sum(tf.transpose(negLogProbs.stack()), -1)

    def Entropy(self, logits):
        entropies = tf.TensorArray(tf.float32, size=tf.shape(logits)[1])
        for i in tf.range(tf.shape(logits)[1]):
            # [batch, ]
            entropy = self._categorical.Entropy(logits[:, i])
            entropies = entropies.write(i, entropy)
        #      [batch, ]
        return tf.reduce_sum(tf.transpose(entropies.stack()), -1)
