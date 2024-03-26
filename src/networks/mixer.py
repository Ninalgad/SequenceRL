import numpy as np
import tensorflow as tf


class MlpMixerNetwork(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=256):
        """MLP-Mixer: An all-MLP Architecture for Vision (https://arxiv.org/pdf/2105.01601.pdf)"""
        super(MlpMixerNetwork, self).__init__()

        self.patch_layer = tf.keras.layers.Conv2D(d_model, 2, strides=2)
        self.vec_emb = tf.keras.layers.Dense(d_model)
        self.encoder = Encoder(num_layers, d_model, dff, 26)
        self.final_layer = point_wise_feed_forward_network(1, dff)

    def call(self, board, vec, training):
        b = self.patch_layer(board)
        b = tf.split(b, 5, axis=1)
        b = [tf.squeeze(x, 1) for x in b]
        b = tf.concat(b, axis=1)

        # use vec features as the first token, bert style
        v = self.vec_emb(vec)
        v = tf.expand_dims(v, 1)

        enc_output = self.encoder(tf.concat([b, v], axis=1), training)  # (batch_size, inp_seq_len, d_model)

        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(enc_output)
        final_output = self.final_layer(pooled_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.keras.activations.tanh(final_output)

        return final_output


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


class MixerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, seq_len=None, transpose=False):
        super(MixerLayer, self).__init__()
        if seq_len is None:
            seq_len = d_model
        self.transpose = transpose
        self.d_model = d_model

        self.norm = tf.keras.layers.LayerNormalization()
        self.mlp = point_wise_feed_forward_network(seq_len, dff)

    def call(self, x):
        x = self.norm(x)
        if self.transpose:
            x = tf.transpose(x, (0, 2, 1))

        x = self.mlp(x)

        if self.transpose:
            x = tf.transpose(x, (0, 2, 1))

        return x


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=tf.nn.gelu),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, seq_len=None):
        super(EncoderLayer, self).__init__()
        self.mixer1 = MixerLayer(d_model, dff, seq_len, True)
        self.mixer2 = MixerLayer(d_model, dff)

    def call(self, x):
        out1 = self.mixer1(x)
        out2 = self.mixer2(out1 + x)
        return out2 + out1


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, dff, seq_len):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(seq_len, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, dff, seq_len)
                           for _ in range(num_layers)]

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)
