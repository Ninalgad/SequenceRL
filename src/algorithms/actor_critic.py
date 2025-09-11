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
    tf.TensorSpec(shape=(1, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(1, 178), dtype=tf.float32, name='vec')
]


def softmax_2d(x):
    b, w, h = x.shape
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax()(x)
    x = tf.reshape(x, (-1, w, h))
    return x


class A2CAlgorithm(Algorithm):
    def __init__(self, init_model, learning_rate=1e-5):
        super(A2CAlgorithm, self).__init__(init_model=init_model, learning_rate=learning_rate)

    def build(self):
        self.model(np.zeros((1, 10, 10, 2)), np.zeros((1, 178)))

    @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    def train_step(self, board, vec, action, tar):
        with tf.GradientTape() as tape:
            qval, policy_logits = self.model(board, vec)
            q_loss = tf.keras.losses.mae(qval, tar)

            policy = softmax_2d(policy_logits)
            advantage = tf.stop_gradient(qval - tar)[:, None]  # (b,1,1)
            a_loss = -advantage * tf.math.log(policy + 1e-10) * action
            a_loss = tf.reduce_sum(a_loss, (1, 2))

            loss = q_loss + a_loss

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))
        self.train_loss(loss)
        return loss

    @tf.function(input_signature=POLICY_SIGNATURE)
    def _policy(self, board, vec):
        q, l = self.model(board, vec)
        return q, l

    def policy(self, board, vec):
        q, l = self.model(board[None], vec[None])
        return q[0].numpy(), l[0].numpy()
