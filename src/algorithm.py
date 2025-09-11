import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):
    def __init__(self, init_model, learning_rate):
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = init_model

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train_step(self, board, vec, action, tar):
        pass

    @abstractmethod
    def policy(self, board_inp, vec_inp):
        pass
