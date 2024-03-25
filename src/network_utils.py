import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(ConvLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(d_model, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(d_model, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.skip_layer = tf.keras.layers.Conv2D(d_model, 1)

    def get_config(self):
        return {}

    def call(self, x):
        skip = self.skip_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + skip

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers):
        super(ConvBlock, self).__init__()
        self.conv_layers = [ConvLayer(d_model) for _ in range(num_layers)]
        self.inp_layer = tf.keras.layers.Conv2D(d_model, 1)

    def get_config(self):
        return {}

    def call(self, x):
        x = self.inp_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ConvEncoder(tf.keras.Model):
    def __init__(self, d_model):
        super(ConvEncoder, self).__init__()
        self.conv1 = ConvBlock(int(d_model // 2), num_layers=3)
        self.conv2 = ConvBlock(d_model, num_layers=3)
        self.output_layer = point_wise_feed_forward_network(d_model, d_model)

    def get_config(self):
        return {}

    def call(self, x):
        x = self.conv1(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = self.conv2(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.output_layer(x)

        return x
