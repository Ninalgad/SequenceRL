import numpy as np
import tensorflow as tf


class VitNetwork(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, num_heads=4, dff=256,
                 rate=0.1):
        super(VitNetwork, self).__init__()

        self.patch_layer = tf.keras.layers.Conv2D(d_model, 2, strides=2)
        self.vec_emb = tf.keras.layers.Dense(d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 26, rate)
        self.final_layer = AttentionPool(output_dim=1, d_model=d_model)

    def preprocess_inp(self, b, v):
        b = self.patch_layer(b)
        b = tf.split(b, 5, axis=1)
        b = [tf.squeeze(x, 1) for x in b]
        b = tf.concat(b, axis=1)

        # use vec features as the first token, bert style
        v = tf.expand_dims(v, 1)

        x = tf.concat([b, v], axis=1)
        return x

    def call(self, board, vec, training):
        v = self.vec_emb(vec)
        x = self.preprocess_inp(board, v)

        enc_output = self.encoder(x, training)

        final_output, _ = self.final_layer(enc_output)
        final_output = tf.keras.activations.tanh(final_output)

        return final_output


class URBEVitNetwork(VitNetwork):
    def __init__(self, num_layers=6, d_model=256, num_heads=4, dff=256,
                 rate=0.1):
        super(URBEVitNetwork, self).__init__(num_layers=num_layers,
        d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)

        self.final_layer = AttentionPool(output_dim=2, d_model=d_model)

    def call(self, board, vec, training):
        v = self.vec_emb(vec)
        x = self.preprocess_inp(board, v)

        enc_output = self.encoder(x, training)

        output, _ = self.final_layer(enc_output)
        q_output, y_output = output[:, :1], output[:, 1:]
        q_output = tf.keras.activations.tanh(q_output)

        return q_output, y_output


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        heads = tf.split(x, self.num_heads, axis=-1)
        x = tf.stack(heads, axis=1)
        return x

    def call(self, v, k, q, mask, training=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class AttentionPool(tf.keras.layers.Layer):
    def __init__(self, output_dim, d_model):
        super(AttentionPool, self).__init__()
        self.num_heads = 2
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(output_dim)

    def split_heads(self, x):
        heads = tf.split(x, self.num_heads, axis=-1)
        x = tf.stack(heads, axis=1)
        return x

    def call(self, x, mask=None):
        v, k, q = x, x, x[:, :1]
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, 1, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q)  # (batch_size, 1, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, 1, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, 1, d_model)

        output = self.dense(concat_attention)  # (batch_size, 1, d_model)
        output = tf.squeeze(output, 1)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class ReZero(tf.keras.layers.Layer):
    def __init__(self, name):
        super(ReZero, self).__init__(name=name)
        a_init = tf.zeros_initializer()
        self.alpha = tf.Variable(name=self.name + '-alpha',
                                 initial_value=a_init(shape=(1,), dtype="float32"), trainable=True
                                 )

    def call(self, inputs):
        x, f_x = inputs
        return x + self.alpha * f_x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.rezero1 = ReZero(self.name + 're0-e-1')
        self.rezero2 = ReZero(self.name + 're0-e-2')

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.rezero1([x, attn_output])  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.rezero2([out1, ffn_output])  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
