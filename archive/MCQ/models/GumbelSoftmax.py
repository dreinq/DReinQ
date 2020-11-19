import tensorflow as tf
import tensorflow_probability as tfp

def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0., maxval=1.)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax(logits, temperature):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    k = tf.shape(logits)[-1]
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
    y_onehot = tf.stop_gradient(y_hard - y) + y

    return y, y_onehot

class FC(tf.keras.Model):
    def __init__(self, ):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(4096, activation=None)

    def call(self, x, training=False):
        # x = self.bn(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x, training=True)
        return self.dense4(x)

class GumbelSoftmax(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.temperature = tf.Variable(0.5, trainable=True)

    def call(self, logits):
        y, y_onehot = gumbel_softmax(logits, self.temperature)
        return y, y_onehot


class Codebook(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # w_init = tf.random_normal_initializer()
        self.w = tf.keras.layers.Dense(128, use_bias=True)
        self.gs = GumbelSoftmax()
        self.klLoss = tf.keras.losses.KLDivergence()

    def call(self, y):
        # y, y_onehot = self.gs(x)
        # idx = tf.argmax(x, -1)

        soft = self.w(y)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y_onehot = tf.stop_gradient(y_hard - y) + y
        hard = self.w(y_onehot)

        self.add_loss(0.1 * self.klLoss(y, y_onehot))
        return soft, hard

class MCQ(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = FC()
        self.fc2 = FC()
        self.fc3 = FC()
        self.fc4 = FC()

        self.codebook1 = Codebook()
        self.codebook2 = Codebook()
        self.codebook3 = Codebook()
        self.codebook4 = Codebook()
        self.mse = tf.keras.losses.MeanSquaredError()
        # self.klLoss = tf.keras.losses.KLDivergence()

    def call(self, x):
        # distance1 = tf.reduce_sum((x[:, None, :] - self.codebook1.w.get_weights()[0][None, :, :])**2, axis=-1)
        # distance2 = tf.reduce_sum((x[:, None, :] - self.codebook2.w.get_weights()[0][None, :, :])**2, axis=-1)
        # distance3 = tf.reduce_sum((x[:, None, :] - self.codebook3.w.get_weights()[0][None, :, :])**2, axis=-1)
        # distance4 = tf.reduce_sum((x[:, None, :] - self.codebook4.w.get_weights()[0][None, :, :])**2, axis=-1)
        logits1, logits2, logits3, logits4 = self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x)
        soft1, hard1 = self.codebook1(logits1)
        soft2, hard2 = self.codebook2(logits2)
        soft3, hard3 = self.codebook3(logits3)
        soft4, hard4 = self.codebook4(logits4)
        soft, hard = tf.add_n([soft1, soft2, soft3, soft4]), tf.add_n([hard1, hard2, hard3, hard4])
        self.add_loss(tf.reduce_mean(tf.reduce_sum((x-hard)**2, axis=-1)))
        self.add_loss(0.1 * tf.reduce_mean(tf.reduce_sum((x - soft) ** 2, axis=-1)))
        # self.add_loss(0.1 * tf.reduce_mean(tf.reduce_sum((soft-hard)**2, axis=-1)))
        return soft # , hard