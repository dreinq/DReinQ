import tensorflow as tf

from MCQ.Consts import Consts

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, numHeads: int):
        """Multi head attention.

        Args:

            numHeads (int): Number of heads.
            logitsOnly (bool, optional): Only return logits for query @ key. Defaults to False.
        """
        super().__init__()
        self._numHeads = numHeads

    def build(self, input_shape: tf.TensorShape):
        self._featureDim = input_shape[-1]
        tf.assert_equal(self._featureDim % self._numHeads, 0)
        self._depth = self._featureDim // self._numHeads
        self._wq = tf.keras.layers.Dense(self._featureDim)
        self._wk = tf.keras.layers.Dense(self._featureDim)
        self._wv = tf.keras.layers.Dense(self._featureDim)
        self._dense = tf.keras.layers.Dense(self._featureDim)

    def _splitHeads(self, x, batch_size):
        """ 分拆最后一个维度到 (_numHeads, depth).

            转置结果使得形状为 (batch_size, _numHeads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self._numHeads, self._depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def _scaledDotProductAttention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor = None):
        """ 计算注意力权重。

            q, k, v 必须具有匹配的前置维度。

            k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。

            虽然 mask 根据其类型（填充或前瞻）有不同的形状，但是 mask 必须能进行广播转换以便求和。

            参数:

                q: 请求的形状 == (..., seq_len_q, depth)
                k: 主键的形状 == (..., seq_len_k, depth)
                v: 数值的形状 == (..., seq_len_v, depth_v)
                mask: Float 张量，其形状能转换成
                    (..., seq_len_q, seq_len_k)。默认为None。

            返回值:
                输出，注意力权重
        """

        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # 缩放 matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor = None, training: bool = False):
        batch_size = tf.shape(query)[0]

        # (batch_size, seq_len, d_model)
        q = self._wq(query)
        # (batch_size, seq_len, d_model)
        k = self._wk(key)
        if self._wv is not None:
            # (batch_size, seq_len, d_model)
            v = self._wv(value)

        # (batch_size, _numHeads, seq_len_q, depth)
        q = self._splitHeads(q, batch_size)
        # (batch_size, _numHeads, seq_len_k, depth)
        k = self._splitHeads(k, batch_size)
        # (batch_size, _numHeads, seq_len_v, depth)
        v = self._splitHeads(v, batch_size)

        # (batch_size, _numHeads, seq_len_q, depth), (batch_size, _numHeads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self._scaledDotProductAttention(q, k, v, mask)

        # (batch_size, seq_len_q, _numHeads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self._featureDim))
        # (batch_size, seq_len_q, d_model)
        output = self._dense(concat_attention)

        return output, attention_weights


def PointwiseFeedforwardNetwork(featureDim, hiddenDim):
    return tf.keras.Sequential([
        # (batch_size, seq_len, hiddenDim)
        tf.keras.layers.Dense(hiddenDim, activation='relu'),
        # (batch_size, seq_len, featureDim)
        tf.keras.layers.Dense(featureDim)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, numHeads: int, hiddenDim: int, rate: float = 0.1):
        """Encoder layer for Transformer.

        Args:

            numHeads (int): Number of heads for MultiHeadAttention.
            hiddenDim (int): Hidden layer dim for PointwiseFeedfowardNetwork.
            rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self._numHeads = numHeads
        self._hiddenDim = hiddenDim

        self._layerNorms = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ]
        self._dropouts = [
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dropout(rate)
        ]

    def build(self, input_shape: tf.TensorShape):
        featureDim = input_shape[-1]
        self._multiHeadAttention = MultiHeadAttention(self._numHeads)
        self._feedfowardNetwork = PointwiseFeedforwardNetwork(featureDim, self._hiddenDim)

    def call(self, x: tf.Tensor, mask: tf.Tensor, training: bool = False):
        # (batch_size, input_seq_len, featureDim)
        attnOutput, _ = self._multiHeadAttention(x, x, x, mask)
        attnOutput = self._dropouts[0](attnOutput, training=training)
        # (batch_size, input_seq_len, featureDim)
        attnOutput = self._layerNorms[0](x + attnOutput)

        # (batch_size, input_seq_len, featureDim)
        ffnOutput = self._feedfowardNetwork(attnOutput)
        ffnOutput = self._dropouts[1](ffnOutput, training=training)
        # (batch_size, input_seq_len, featureDim)
        return self._layerNorms[1](attnOutput + ffnOutput)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, numHeads: int, hiddenDim: int, rate: float = 0.1):
        """Decoder layer for Transformer.

        Args:

            numHeads (int): Number of heads for MultiHeadAttention.
            hiddenDim (int): Hidden layer dim for PointwiseFeedfowardNetwork.
            rate (float, optional): Dropout rate. Defaults to 0.1.
            logitsOnly (bool, optional): Should only return logits (query @ key) for attentions. Defaults to False.
        """
        super().__init__()

        self._numHeads = numHeads
        self._hiddenDim = hiddenDim

        self._layerNorms = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ]

        self._dropouts = [
            tf.keras.layers.Dropout(rate)
        ]
        self._dropouts.append(tf.keras.layers.Dropout(rate))
        self._layerNorms.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
        self._dropouts.append(tf.keras.layers.Dropout(rate))
        self._layerNorms.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def build(self, input_shape: tf.TensorShape):
        featureDim = input_shape[-1]
        self._multiHeadAttentions = [
            MultiHeadAttention(self._numHeads),
            MultiHeadAttention(self._numHeads)
        ]
        self._feedfowardNetwork = PointwiseFeedforwardNetwork(featureDim, self._hiddenDim)

    def call(self, x: tf.Tensor, encoderOutput: tf.Tensor, lookAheadMask: tf.Tensor, paddingMask: tf.Tensor, training: bool = False):
        # encoderOutput.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, _ = self._multiHeadAttentions[0](x, x, x, lookAheadMask, training)
        attn1 = self._dropouts[0](attn1, training)

        out1 = attn1
        out1 = self._layerNorms[0](attn1 + x)

        # (batch_size, target_seq_len, d_model)
        attn2, _ = self._multiHeadAttentions[1](out1, encoderOutput, encoderOutput, paddingMask, training)
        # if self._logitsOnly:
        #     return None, attn_weights_block2
        attn2 = self._dropouts[1](attn2, training)
        # (batch_size, target_seq_len, d_model)
        out2 = self._layerNorms[1](attn2 + out1)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self._feedfowardNetwork(out2)
        ffn_output = self._dropouts[2](ffn_output, training)
        # (batch_size, target_seq_len, d_model)
        out3 = self._layerNorms[2](ffn_output + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self, numLayers: int, numHeads: int, hiddenDim: int, rate: float = 0.1):
        super(Encoder, self).__init__()

        self._numLayers = numLayers

        self._encoderLayers = [
            EncoderLayer(numHeads, hiddenDim, rate)
            for _ in range(numLayers)
        ]
        # self._dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask = None, training = False):
        # x = self._dropout(x, training)

        for i in range(self._numLayers):
            x = self._encoderLayers[i](x, mask, training)

        # (batch_size, input_seq_len, featureDim)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, numLayers, numCodebooks, numCodewords, numHeads, hiddenDim, rate=0.1):
        super().__init__()

        self._numLayers = numLayers
        self._numCodebooks = numCodebooks
        self._numCodewords = numCodewords

        self._decoderLayers = [
            DecoderLayer(numHeads, hiddenDim, rate)
            for i in range(numCodebooks)
        ]
        # self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        pass
        # self._xTwistedInitialSequence = self.add_weight(name="InitialSequence", shape=[1, 1, 2 * input_shape[-1]], initializer="random_normal")

    @staticmethod
    def _lookAhead(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        # (seq_len, seq_len)
        return mask

    def call(self, x, encoderOutput, training):
        # [batch, 1, d]
        xInput = tf.expand_dims(x, 1)
        xFeed = xInput
        # [batch, 1, 3d] <- [batch, 1, d] - [1, 1, 2d]
        # xInput = tf.concat([xSequenced, tf.broadcast_to(self._xTwistedInitialSequence, [xSequenced.shape[0], 1, xSequenced.shape[-1]])], 1)
        # codebookLogits = tf.TensorArray(tf.float32, size=self._numCodebooks)
        for m in range(self._numCodebooks):
            lookAheadMask = self._lookAhead(tf.shape(xFeed)[1])
            for i in range(self._numLayers):
                # [batch, m, d]
                xFeed = self._decoderLayers[i](xFeed, encoderOutput, lookAheadMask, None, training)
                # xSequenced = tf.concat([xSequenced, xTwisted], 1)
                # codebookLogits.write(i, codewordLogits)
            if m < self._numCodebooks - 1:
                # [batch, m+1, d]
                xFeed = tf.concat([xInput, xFeed], axis=1)
        return xFeed
        # if training:
        #     quantized = tf.zeros([x.shape[0], num_samples, x.shape[-1]])
        # else:
        #     quantized = None
        # x_sequenced = tf.expand_dims(x, 1)
        # cumulative = tf.zeros(x.shape)
        # context_node = tf.concat([x_sequenced, self.init_value, cumulative[:, None, ...]], -1)

        # pi = list()

        # partialMask = tf.zeros([1, self.numCodewords])

        # left = 0
        # right = (self.numCodebooks-1)*self.numCodewords

        # codewordMask = tf.pad(partialMask, [[0, 0], [left, right]], constant_values=1.)
        # assert codewordMask.shape[-1] == self.numCodebooks * self.numCodewords

        # for i in range(self.numCodebooks):
        #     x_twisted, codeword_logits = self.dec_layers[i](
        #         context_node, enc_output, training, None,
        #         codewordMask, logits_only=i==self.numCodebooks-1)
        #     codeword_logits = tf.reshape(codeword_logits, [x.shape[0], -1])
        #     """ ************************** """
        #     if training:
        #         thisCodebook_p = codeword_logits[..., self.numCodewords*i:self.numCodewords*(i+1)]
        #         # log_probability = tf.math.log(codeword_probability + 1e-10)
        #         # thisCodebook_p = codeword_probability[..., self.numCodewords*i:self.numCodewords*(i+1)]
        #         # [batch, num_samples]
        #         codeword, picked_p = self.pick_codeword_by_categorical(codebook[:, self.numCodewords*i:self.numCodewords*(i+1)], thisCodebook_p, num_samples)
        #         quantized += codeword
        #         cumulative += tf.reduce_mean(codeword, 1)

        #     else:
        #         thisCodebook_p = self.softmax(codeword_logits[..., self.numCodewords*i:self.numCodewords*(i+1)])
        #         # log_probability = tf.math.log(codeword_probability + 1e-10)
        #         # thisCodebook_p = codeword_probability[..., self.numCodewords*i:self.numCodewords*(i+1)]
        #         codeword, picked_p = self.pick_codeword_from_wo_gumbel(codebook[:, self.numCodewords*i:self.numCodewords*(i+1)], thisCodebook_p)
        #         cumulative += codeword

        #     log_probability = tf.math.log(picked_p)
        #     pi.append(log_probability)
        #     """ ************************ """
        #     if i == self.numCodebooks - 1:
        #         break
        #     context_node = tf.concat([x_sequenced, x_twisted, cumulative[:, None, ...]], -1)
        #     left += self.numCodewords
        #     right -= self.numCodewords
        #     codewordMask = tf.pad(partialMask, [[0, 0], [left, right]], constant_values=1.)
        #     assert codewordMask.shape[-1] == self.numCodebooks * self.numCodewords

        #     if not training:
        #         quantized = cumulative

        # # x.shape == (batch_size, target_seq_len, d_model)
        # return pi, quantized

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
        ix = tf.range(tf.shape(y_hard)[0], dtype=y_hard.dtype)
        # [batch, num_samples, 1]
        ix = tf.repeat(ix[:, None, None], [num_samples], axis=1)
        # [batch, num_samples, 2]
        y_hard = tf.concat([ix, y_hard], -1)
        probability = tf.nn.softmax(p)
        # direct gather by coordinates
        # [batch, K, D] gather by [batch, num_samples, 2],,, [batch, K] gather by [batch, num_samples, 2]
        return tf.gather_nd(subCodebook, y_hard), tf.gather_nd(probability, y_hard)


class Sampler(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, subCodebook, numSamples, logits):
        # [batch, num_samples, 1]
        samplesY = tf.random.categorical(logits, numSamples)[..., None]
        # [batch]
        ix = tf.range(tf.shape(samplesY)[0], dtype=samplesY.dtype)
        # [batch, num_samples, 1]
        ix = tf.broadcast_to(ix[:, None, None], tf.shape(samplesY))
        # [batch, num_samples, 2]
        indices = tf.concat([ix, samplesY], -1)
        # softmax probability
        probs = tf.nn.softmax(logits)
        # direct gather by coordinates
        # [batch, K, D] gather by [batch, num_samples, 2],,, [batch, K] gather by [batch, num_samples, 2]
        return tf.gather_nd(subCodebook, indices), tf.gather_nd(probs, indices)

class GreedyPicker(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, subCodebook, logits):
        y_hard = tf.argmax(logits, -1)[:, None]
        return tf.gather_nd(subCodebook, y_hard, batch_dims=1)

class Threshold(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, value):
        scale = tf.cast(tf.shape(value)[-1], tf.float32)
        # return (tf.stop_gradient(tf.nn.tanh(value) - value) + value) * tf.math.log(scale)
        return tf.nn.tanh(value) * tf.math.log(scale)

class Noiser(tf.keras.layers.Layer):
    def __init__(self, mean, var):
        super().__init__()
        self._mean = mean
        self._var = var
        self._eps = 1e-15

    def call(self, value):
        U = tf.random.uniform(tf.shape(value), minval=0., maxval=1.)
        return value + -1. * tf.math.log(-1. * tf.math.log(U + self._eps) + self._eps)
        # noise = tf.random.normal(tf.shape(value), mean=self._mean, stddev=self._var)
        # return value + noise

class Transformer(tf.keras.Model):
    def __init__(self, numLayers, numCodebooks, numCodewords, numHeads, hiddenDim, rate=0.1):
        super(Transformer, self).__init__()
        self._numCodebooks = numCodebooks
        self._encoder = Encoder(numLayers, numHeads, hiddenDim, rate)
        self._decoder = Decoder(numLayers, numCodebooks, numCodewords, numHeads, hiddenDim, rate)
        self._sampler = Sampler()
        self._greedyPicker = GreedyPicker()
        self._threshold = Threshold()
        self._finalLayer = tf.keras.layers.Dense(numCodewords, activity_regularizer=tf.keras.regularizers.l2(0.01))

    def build(self, input_shape):
        self._noiser = Noiser(0.0, tf.math.log(tf.cast(input_shape[-1], tf.float32)))

    def call(self, x, codebook, temperature, numSamples, training):
        flattenCodebook = tf.reshape(codebook, [tf.shape(x)[0], -1, x.shape[-1]])
        # (batch, M * K, D)
        encoderOutput = self._encoder(flattenCodebook, training=training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        # pi: list of log codeword prbability [(batch, K) * M]
        # quantized: (batch, D)

        # [batch, M, D]
        xDecoded = self._decoder(x, encoderOutput, training)

        # [batch, M, K], predict logits for all codewords
        codebookLogits = self._finalLayer(xDecoded)

        quantized = tf.TensorArray(tf.float32, size=self._numCodebooks, dynamic_size=False)
        probs = tf.TensorArray(tf.float32, size=self._numCodebooks, dynamic_size=False)

        if training:
            # codebookLogits /= tf.math.sqrt(scale)
            # codebookLogits = self._noiser(codebookLogits)
            codebookLogits = self._threshold(codebookLogits)
            # codebookLogits /= tf.broadcast_to(temperature, tf.shape(codebookLogits))

        for i in tf.range(self._numCodebooks):
            if training:
                # scale = tf.cast(tf.shape(codebookLogits)[-1], tf.float32)
                # [batch, numSamples, D], [batch, numSamples]
                pickedCodeword, corrProb = self._sampler(codebook[:, i], numSamples, codebookLogits[:, i])
                quantized = quantized.write(i, pickedCodeword)
                probs = probs.write(i, corrProb)
            else:
                # [batch, D]
                pickedCodeword = self._greedyPicker(codebook[:, i], codebookLogits[:, i])
                quantized = quantized.write(i, pickedCodeword)
        # [batch, numSamples, D] or [batch, D] <- [M, batch, numSamples, D] or [M, batch, D]
        quantized = tf.reduce_sum(quantized.stack(), 0)

        if training:
            # [batch, numSamples, M]
            probs = tf.transpose(probs.stack(), (1, 2, 0))
            # [batch, numSamples]
            probs = tf.reduce_prod(probs, -1)
            logProbs = tf.math.log(probs)
            return quantized, logProbs

        return quantized
