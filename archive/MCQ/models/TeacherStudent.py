import tensorflow as tf
import tensorflow_probability as tfp

def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0., maxval=1.)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax(logits, temperature):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
    y_onehot = tf.stop_gradient(y_hard - y) + y

    return y, y_onehot

class FC(tf.keras.Model):
    def __init__(self, last_acti=None):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.1)
        # self.bn = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(256, activation=last_acti)

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

    def call(self, logits, temperature):
        y, y_onehot = gumbel_softmax(logits, temperature)
        return y, y_onehot


class Codebook(tf.keras.layers.Layer):
    def __init__(self, soft=False):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init([256, 128]), trainable=True, name="Codebook")
        # self.scale = tf.Variable(initial_value=1.0, trainable=True, name="Scale")
        # self.w = tf.keras.layers.Dense(128, use_bias=False)
        self._soft = soft
        if not self._soft:
            self.gs = GumbelSoftmax()

    def call(self, y, temperature=0.5):
        # y, y_onehot = self.gs(x)
        # idx = tf.argmax(x, -1)
        if self._soft:
            return y @ self.w
        y_sample, y_onehot = self.gs(y, temperature)
        # y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        # y_onehot = tf.stop_gradient(y_hard - y) + y
        # tf.print(tf.argmax(y_onehot, axis=-1)[0])
        return y_onehot @ self.w, y_sample, y_onehot

class Teacher(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = FC(tf.nn.softmax)
        self.fc2 = FC(tf.nn.softmax)
        self.fc3 = FC(tf.nn.softmax)
        self.fc4 = FC(tf.nn.softmax)

        self.codebook1 = Codebook(soft=False)
        self.codebook2 = Codebook(soft=False)
        self.codebook3 = Codebook(soft=False)
        self.codebook4 = Codebook(soft=False)
        self.codebook = [self.codebook1, self.codebook2, self.codebook3, self.codebook4]

    def call(self, x, reference=False, temperature=0.5):
        # distance1 = tf.reduce_sum((x[:, None, :] - self.codebook1.w[None, :, :])**2, axis=-1)
        # distance2 = tf.reduce_sum((x[:, None, :] - self.codebook2.w[None, :, :])**2, axis=-1)
        # distance3 = tf.reduce_sum((x[:, None, :] - self.codebook3.w[None, :, :])**2, axis=-1)
        # distance4 = tf.reduce_sum((x[:, None, :] - self.codebook4.w[None, :, :])**2, axis=-1)
        logits1, logits2, logits3, logits4 = self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x)
        hard1, logits1, code1 = self.codebook1(logits1, temperature)
        hard2, logits2, code2 = self.codebook2(logits2, temperature)
        hard3, logits3, code3 = self.codebook3(logits3, temperature)
        hard4, logits4, code4 = self.codebook4(logits4, temperature)
        hard = tf.add_n([hard1, hard2, hard3, hard4])
        # soft = tf.add_n()
        # self.add_loss(tf.reduce_mean(tf.reduce_sum((x - soft) ** 2, axis=-1)))
        # self.add_loss(0.1 * tf.reduce_mean(tf.reduce_sum((soft-hard)**2, axis=-1)))
        return hard, (logits1, code1), (logits2, code2), (logits3, code3), (logits4, code4)

    @property
    def codebookWeights(self):
        return self.codebook1.trainable_weights + self.codebook2.trainable_weights + self.codebook3.trainable_weights + self.codebook4.trainable_weights

    @property
    def fcWeights(self):
        return self.fc1.trainable_weights + self.fc2.trainable_weights + self.fc3.trainable_weights + self.fc4.trainable_weights

class Assistant(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = FC(tf.nn.softmax)
        self.fc2 = FC(tf.nn.softmax)
        self.fc3 = FC(tf.nn.softmax)
        self.fc4 = FC(tf.nn.softmax)

        self.codebook1 = Codebook(soft=True)
        self.codebook2 = Codebook(soft=True)
        self.codebook3 = Codebook(soft=True)
        self.codebook4 = Codebook(soft=True)
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, x):
        logits1, logits2, logits3, logits4 = self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x)
        hard1 = self.codebook1(logits1)
        hard2 = self.codebook2(logits2)
        hard3 = self.codebook3(logits3)
        hard4 = self.codebook4(logits4)
        hard = tf.add_n([hard1, hard2, hard3, hard4])
        # code = tf.concat(values=[code1, code2, code3, code4], axis=-1)
        # self.add_loss(tf.reduce_mean(tf.reduce_sum((hard - softY) ** 2, axis=-1)))
        # self.add_loss(0.1 * tf.reduce_mean(tf.reduce_sum((soft-hard)**2, axis=-1)))
        return hard, hard1, hard2, hard3, hard4

class TeacherStudent(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher()
        self.assistant = Assistant()
        # self.student = Student()
        # self.klLoss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        # self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, x):
        soft = self.teacher(x)
        # hard, code = self.student(x)
        # self.add_loss(self.klLoss(soft, hard))
        # self.add_loss(self.mse(soft, hard))
        # self.add_loss(0.1 * tf.reduce_mean(tf.reduce_sum((soft - hard) ** 2, axis=-1)))
        return soft # , hard, code
