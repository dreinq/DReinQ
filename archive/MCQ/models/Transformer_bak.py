import os
from typing import Type
from random import shuffle

import tensorflow as tf

from MCQ.Consts import Consts


def scaled_dot_product_attention(q, k, v, mask=None, raw_logits=False, logits_only=False):
    """ 计算注意力权重。
        q, k, v 必须具有匹配的前置维度。
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。
        
        参数:
            q: 请求的形状 == (..., seq_len_q, depth)
            k: 主键的形状 == (..., seq_len_k, depth)
            v: 数值的形状 == (..., seq_len_v, depth_v)
            mask: Float 张量，其形状能转换成
                (..., seq_len_q, seq_len_k)。默认为None。
            
        返回值:
            输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k,
                          transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    if logits_only:
        return None, scaled_attention_logits

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    if raw_logits:
        return output, scaled_attention_logits
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, logitsOnly=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)

        if not logitsOnly:
            self.wv = tf.keras.layers.Dense(d_model)
            self.dense = tf.keras.layers.Dense(d_model)
        else:
            self.wv = None

    def split_heads(self, x, batch_size):
        """ 分拆最后一个维度到 (num_heads, depth).
            转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, raw_logits=False, logits_only=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        if self.wv is not None:
            v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(
            q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, raw_logits, logits_only)

        if logits_only:
            return None, attention_weights

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(
            concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,
                              activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x,
                                  mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, logitsOnly=False):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads, logitsOnly)


        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        if not logitsOnly:
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.dropout3 = tf.keras.layers.Dropout(rate)
            self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, logits_only=False):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, _ = self.mha1(
            x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)

        out1 = attn1
        # out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1,
            padding_mask, raw_logits=True, logits_only=logits_only)  # (batch_size, target_seq_len, d_model)
        if logits_only:
            return None, attn_weights_block2
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 +
                               out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output +
                               out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, numCodebooks, numCodewords, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.numCodebooks = numCodebooks
        self.numCodewords = numCodewords

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate, i==numCodebooks-1)
            for i in range(numCodebooks)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.init_value = tf.Variable(initial_value=w_init(input_shape),
                                      trainable=None)[:, None, ...]

    def call(self, x, codebook, enc_output, gumbel_temp, num_samples, training):
        if training:
            quantized = tf.zeros([x.shape[0], num_samples, x.shape[-1]])
        else:
            quantized = None
        x_sequenced = tf.expand_dims(x, 1)
        cumulative = tf.zeros(x.shape)
        context_node = tf.concat([x_sequenced, self.init_value, cumulative[:, None, ...]], -1)

        pi = list()

        partialMask = tf.zeros([1, self.numCodewords])

        left = 0
        right = (self.numCodebooks-1)*self.numCodewords

        codewordMask = tf.pad(partialMask, [[0, 0], [left, right]], constant_values=1.)
        assert codewordMask.shape[-1] == self.numCodebooks * self.numCodewords

        for i in range(self.numCodebooks):
            x_twisted, codeword_logits = self.dec_layers[i](
                context_node, enc_output, training, None,
                codewordMask, logits_only=i==self.numCodebooks-1)
            codeword_logits = tf.reshape(codeword_logits, [x.shape[0], -1])
            """ ************************** """
            if training:
                thisCodebook_p = codeword_logits[..., self.numCodewords*i:self.numCodewords*(i+1)]
                # log_probability = tf.math.log(codeword_probability + 1e-10)
                # thisCodebook_p = codeword_probability[..., self.numCodewords*i:self.numCodewords*(i+1)]
                # [batch, num_samples]
                codeword, picked_p = self.pick_codeword_by_categorical(codebook[:, self.numCodewords*i:self.numCodewords*(i+1)], thisCodebook_p, num_samples)
                quantized += codeword
                cumulative += tf.reduce_mean(codeword, 1)

            else:
                thisCodebook_p = self.softmax(codeword_logits[..., self.numCodewords*i:self.numCodewords*(i+1)])
                # log_probability = tf.math.log(codeword_probability + 1e-10)
                # thisCodebook_p = codeword_probability[..., self.numCodewords*i:self.numCodewords*(i+1)]
                codeword, picked_p = self.pick_codeword_from_wo_gumbel(codebook[:, self.numCodewords*i:self.numCodewords*(i+1)], thisCodebook_p)
                cumulative += codeword

            log_probability = tf.math.log(picked_p)
            pi.append(log_probability)
            """ ************************ """
            if i == self.numCodebooks - 1:
                break
            context_node = tf.concat([x_sequenced, x_twisted, cumulative[:, None, ...]], -1)
            left += self.numCodewords
            right -= self.numCodewords
            codewordMask = tf.pad(partialMask, [[0, 0], [left, right]], constant_values=1.)
            assert codewordMask.shape[-1] == self.numCodebooks * self.numCodewords

            if not training:
                quantized = cumulative

        # x.shape == (batch_size, target_seq_len, d_model)
        return pi, quantized

    def clipped_probability(self, logits, mask=None):
        p = tf.nn.tanh(logits)
        p *= 10.0
        if mask is not None:
            p += (mask * -1e9)
        return tf.nn.softmax(p)

    def softmax(self, logits, mask=None):
        if mask is not None:
            logits += (mask * -1e9)
        return tf.nn.softmax(logits)

    def pick_codeword_from(self, subCodebook, logits, temperature):
        def sample_gumbel(shape, eps=1e-20):
            U = tf.random.uniform(shape, minval=0., maxval=1.)
            return -1. * tf.math.log(-1. * tf.math.log(U + eps) + eps)

        def gumbel_softmax(logits, temperature):
            gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
            y = tf.nn.softmax(gumbel_softmax_sample / temperature)
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), y.dtype)
            y_onehot = tf.stop_gradient(y_hard - y) + y
            return y, y_onehot
        gumbel_p, gumbel_sampled = gumbel_softmax(logits, temperature)
        return tf.einsum("ijk,ij->ik", subCodebook, gumbel_sampled), gumbel_p


    def pick_codeword_from_wo_gumbel(self, subCodebook, probability):
        y_hard = tf.argmax(probability, -1)[:, None]
        return tf.gather_nd(subCodebook, y_hard, batch_dims=1), tf.gather_nd(probability, y_hard, batch_dims=1)
        y_hard = tf.cast(tf.equal(probability, tf.reduce_max(probability, -1, keepdims=True)), probability.dtype)
        return y_hard @ subCodebook, y_hard

    def pick_codeword_by_categorical(self, subCodebook, logits, num_samples):
        # p = tf.nn.tanh(logits)
        # p *= 10.0
        p = logits
        # [batch, num_samples, 1]
        y_hard = tf.random.categorical(p, num_samples)[..., None]
        # [batch]
        ix = tf.range(y_hard.shape[0], dtype=y_hard.dtype)
        # [batch, num_samples, 1]
        ix = tf.repeat(ix[:, None, None], [num_samples], axis=1)
        # [batch, num_samples, 2]
        y_hard = tf.concat([ix, y_hard], -1)
        probability = tf.nn.softmax(p)
        # direct gather by coordinates
        # [batch, K, D] gather by [batch, num_samples, 2],,, [batch, K] gather by [batch, num_samples, 2]
        return tf.gather_nd(subCodebook, y_hard), tf.gather_nd(probability, y_hard)


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 num_codebooks,
                 num_codewords,
                 d_model,
                 num_heads,
                 dff,
                 rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.decoder = Decoder(num_codebooks, num_codewords, d_model, num_heads, dff, rate)

    def call(self, codebook, x, gumbel_temp, num_samples, training):

        enc_output = self.encoder(
            codebook, training, None)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        # pi: list of log codeword prbability [(batch, K) * M]
        # quantized: (batch, D)
        pi, quantized = self.decoder(x, codebook, enc_output, gumbel_temp, num_samples, training)

        # final_output = self.final_layer(
        #     dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return pi, quantized


class Estimator(tf.keras.Model):
    def __init__(self, initCodebook=None):
        super().__init__()
        self._m = 4
        self._k = 256
        self.index = list(range(self._m))
        self.initCodebook = initCodebook

    def build(self, input_shape):
        self.model = Transformer(3, self._m, self._k, input_shape[-1], 1, input_shape[-1] * 2)
        if self.initCodebook is None:
            print("Use random init codebook")
            w_init = tf.random_normal_initializer()
            self.codebook = [
                tf.Variable(w_init([self._k, input_shape[-1]]), trainable=True, name=f"codebook{i}")
                for i in range(self._m)
            ]
        else:
            print("Fill codebook with init value")
            self.codebook = [
                tf.Variable(self.initCodebook[i], trainable=True, name=f"codebook{i}")
                for i in range(self._m)
            ]

    def call(self, x, gumbel_temp, num_samples, training=False):
        # pi, quantized = self.model(tf.repeat(tf.concat(self.codebook, 0)[None, ...], repeats=x.shape[0], axis=0), x, gumbel_temp, training)
        # return pi, quantized
        if training:
            shuffledCodebook = []
            for _ in range(x.shape[0]):
                shuffle(self.index)
                shuffledCodebook.append(tf.concat([self.codebook[i] for i in self.index], 0)) # [tf.random.shuffle(self.codebook[i]) for i in self.index], 0))
            shuffledCodebook = tf.convert_to_tensor(shuffledCodebook)
            pi, quantized = self.model(tf.stop_gradient(shuffledCodebook), x, gumbel_temp, num_samples, training)
            return pi, quantized
        else:
            pi, quantized = self.model(tf.repeat(tf.concat(self.codebook, 0)[None, ...], repeats=x.shape[0], axis=0), x, gumbel_temp, num_samples, training)
            return pi, quantized

class Trainer(tf.keras.Model):
    def __init__(self, initCodebook):
        super().__init__()
        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        assert len(self.gpus) == 2
        with tf.device(self.gpus[0].name):
            self.current = Estimator(initCodebook)
        with tf.device(self.gpus[1].name):
            self.baseline = Estimator(initCodebook)
            self.baseline.trainable = False

        self.accumulate_rewards = None
        self._first_pass = False

        self.optimT = tf.keras.optimizers.Adam(0.00001)
        self.optimC = tf.keras.optimizers.SGD(0.01, momentum=0.95)
        self.optimC_baseline = tf.keras.optimizers.SGD(0.01, momentum=0.95)

    # def CloneModel(self, model):
    #     new_model = tf.keras.models.clone_model(model)
    #     new_model.trainable = False
    #     return new_model

    def FirstPass(self, x):
        self._first_pass = True
        with tf.device(self.gpus[0].name):
            _ = self.current(x, 1.0, 16, True)
        with tf.device(self.gpus[1].name):
            _ = self.baseline(x, 1.0, 16, True)

    def TrainTransformer(self, x, gumbel_temp):
        if not self._first_pass:
            self.FirstPass(x)
        with tf.GradientTape() as t:
            with tf.device(self.gpus[0].name):
                pi, quantized = self.current(x, gumbel_temp, 16, True)
                # [batch, num_samples, M]
                pi = tf.stack(pi, -1)
                qe = tf.reduce_sum((x[:, None, ...] - quantized) ** 2, -1)
            with t.stop_recording(), tf.device(self.gpus[1].name):
                _, baseline_q = self.baseline(x, gumbel_temp, 16, False)
                qe_baseline = tf.reduce_sum((x - baseline_q) ** 2, -1)

            # [batch, num_samples] - [batch, 1]
            reward = qe - qe_baseline[:, None]

            # if self.accumulate_rewards is None:
            #     self.accumulate_rewards = tf.reduce_mean(reward)
            # else:d
            #     self.accumulate_rewards -= (1.0 - 0.995) * (self.accumulate_rewards - tf.reduce_mean(reward))

            # if self.accumulate_rewards < -tf.reduce_mean(tf.abs(reward)):
                # print("Now we get a new baseline, since the accumulate quantization error is far lower than baseline")
                # self.baseline.set_weights(self.current.get_weights())
                # self.accumulate_rewards = None

            # print(self.accumulate_rewards)

            # mean of [batch, num_samples, M]
            reinforce = tf.reduce_mean(reward[..., None] * pi)

        grads = t.gradient(reinforce, self.current.model.trainable_weights)
        self.optimT.apply_gradients(zip(grads, self.current.model.trainable_weights))
        return tf.reduce_mean(reward).numpy(), tf.reduce_mean(qe).numpy(), tf.reduce_mean(qe_baseline).numpy()

    def TrainCodebook(self, x, gumbel_temp):
        with tf.device(self.gpus[0].name):
            with tf.GradientTape() as t:
                _, quantized = self.current(x, gumbel_temp, 1, False)
                qe = tf.reduce_mean(tf.reduce_sum((x - quantized) ** 2, -1))
            grads = t.gradient(qe, self.current.codebook)
            self.optimC.apply_gradients(zip(grads, self.current.codebook))
        with tf.device(self.gpus[1].name):
            with tf.GradientTape() as t:
                _, quantized_b = self.baseline(x, gumbel_temp, 1, False)
                qe_b = tf.reduce_mean(tf.reduce_sum((x - quantized_b) ** 2, -1))
            grads = t.gradient(qe_b, self.baseline.codebook)
            self.optimC_baseline.apply_gradients(zip(grads, self.baseline.codebook))
        return qe, qe_b

    def UpdateCodebookOnBaseline(self):
        for i, c in enumerate(self.baseline.codebook):
            c.assign(self.current.codebook[i])

    def Eval(self):
        self.diff = 0.0
        self.count = 0

    def AddNewEvalPair(self, x):
        with tf.device(self.gpus[0].name):
            _, quantized = self.current(x, 1.0, 1, False)
            qe = tf.reduce_mean(tf.reduce_sum((x - quantized) ** 2, -1))
        with tf.device(self.gpus[1].name):
            _, quantized_b = self.baseline(x, 1.0, 1, False)
            qe_b = tf.reduce_mean(tf.reduce_sum((x - quantized_b) ** 2, -1))
        self.diff += tf.reduce_sum(qe - qe_b)
        self.count += 32
        print("diff:", self.diff)

    def CheckBaseline(self):
        if self.diff / self.count < -100.0:
            print("Replace baseline with current model")
            self.baseline.set_weights(self.current.get_weights())
        elif self.diff / self.count > 100.0:
            print("current model need more exploration")
            return True
        return False
