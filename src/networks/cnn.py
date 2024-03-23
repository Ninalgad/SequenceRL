from src.network_utils import *


class ConvNetwork(tf.keras.Model):
    def __init__(self, d_model=128):
        super(ConvNetwork, self).__init__()
        self.board_cnn = ConvEncoder(d_model)
        self.output_layer = point_wise_feed_forward_network(1, d_model)

    def get_config(self):
        return {}

    def call(self, board, vec):
        b = self.board_cnn(board)
        x = tf.concat([b, vec], axis=-1)
        x = self.output_layer(x)
        x = tf.keras.activations.tanh(x)
        return x
