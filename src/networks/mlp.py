import tensorflow as tf


class MLPNetwork(tf.keras.Model):
    def __init__(self, num_blocks=2, d_model=256, dff=256):
        super(MLPNetwork, self).__init__()

        self.patch_layer = tf.keras.layers.Conv2D(d_model, 2, strides=2)
        self.emb = tf.keras.layers.Dense(d_model)
        self.encoder_layers = [EncoderLayer(d_model, dff) for _ in range(num_blocks)]
        self.final_layer = point_wise_feed_forward_network(1, dff)

    @staticmethod
    def preprocess_inp(b, a, v):
        b = tf.keras.layers.Flatten()(b)
        a = tf.keras.layers.Flatten()(a)
        v = tf.keras.layers.Flatten()(v)
        x = tf.concat([b, a, v], axis=1)

        return x

    def call(self, board, act, vec):
        x = self.preprocess_inp(board, act, vec)
        x = self.emb(x)
        for lay in self.encoder_layers:
            x = lay(x)

        x = self.final_layer(x)
        return tf.keras.activations.tanh(x)


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
    def __init__(self, d_model, dff):
        super(EncoderLayer, self).__init__()

        self.mlp = point_wise_feed_forward_network(d_model, dff)
        self.rezero = ReZero(self.name + 're0')

    def call(self, x):
        return self.rezero([self.mlp(x), x])
