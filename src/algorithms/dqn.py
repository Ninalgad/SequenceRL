import tensorflow as tf
import numpy as np
from src.algorithm import Algorithm


TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(None, 178), dtype=tf.float32, name='vec'),
    tf.TensorSpec(shape=(None, 10, 10), dtype=tf.float32, name='action'),
    tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='tar')
]

POLICY_SIGNATURE = [
    tf.TensorSpec(shape=(None, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(None, 10, 10), dtype=tf.float32, name='action'),
    tf.TensorSpec(shape=(None, 178), dtype=tf.float32, name='vec')
]


class DQNAlgorithm(Algorithm):
    def __init__(self, init_model, learning_rate=1e-5):
        super(DQNAlgorithm, self).__init__(init_model=init_model, learning_rate=learning_rate)

    def build(self):
        self.model.build()

    @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    def train_step(self, board, vec, action, tar):
        with tf.GradientTape() as tape:
            pred = self.model(board, action, vec)
            loss = tf.keras.losses.mae(pred, tar)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))
        self.train_loss(loss)
        return loss

    @tf.function(input_signature=POLICY_SIGNATURE)
    def policy(self, board, action, vec):
        q = self.model(board, action, vec)
        return q
