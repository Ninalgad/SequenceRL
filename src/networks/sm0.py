import tensorflow as tf
import numpy as np
from actors.sm0.typers import *
from typers import Action


POLICY_SIGNATURE = [
    tf.TensorSpec(shape=(1, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(1, 178), dtype=tf.float32, name='vec')
]


class PolicyValueNetwork(tf.keras.Model):
    def __init__(self, policy_dim, dff=256):
        super(PolicyValueNetwork, self).__init__()
        self.model = point_wise_feed_forward_network(policy_dim + 1, dff)

    def call(self, state):

        x = self.model(state)
        v = tf.keras.activations.tanh(x[:, :1])
        p = x[:, 1:]

        return p, v


class DynamicsNetwork(tf.keras.Model):
    def __init__(self, d_model, dff=256):
        super(DynamicsNetwork, self).__init__()
        self.model = point_wise_feed_forward_network(d_model + 1, dff)

    def call(self, state, chance):
        x = tf.concat([state, chance], axis=1)
        x = self.model(x)
        v = tf.keras.activations.tanh(x[:, :1])
        s = x[:, 1:]

        return s, v


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def open_policy_map(policy):
    policy = policy.reshape((10, 10))
    norm = np.exp(policy).sum()
    open_policy = {}
    for i in range(10):
        for j in range(10):
            open_policy[(i, j)] = np.exp(policy[i, j]) / norm
    return open_policy


class StochasticMuZeroNetwork(tf.keras.Model):
    """An instance of the network used by stochastic MuZero."""
    def __init__(self, d_model=256, dff=256, num_codes=32):
        super(StochasticMuZeroNetwork, self).__init__()
        self.h = point_wise_feed_forward_network(d_model, dff)
        self.f = PolicyValueNetwork(100, dff)
        self.phi = point_wise_feed_forward_network(d_model, dff)
        self.psi = PolicyValueNetwork(num_codes, dff)
        self.g = PolicyValueNetwork(d_model, dff)
        self.e = point_wise_feed_forward_network(num_codes, dff)
        self.num_codes = num_codes

    @staticmethod
    def preprocess_obs(b, v):
        b = tf.keras.layers.Flatten()(b)
        v = tf.keras.layers.Flatten()(v)
        x = tf.concat([b, v], axis=1)
        return x

    def build(self):
        b, v = np.zeros((1, 10, 10, 2)), np.zeros((1, 178))
        a = np.zeros((1, 100))
        c = np.zeros((1, 32))
        s = self._representation(b, v)
        _ = self._encoder(b, v)
        p, v = self._predictions(s)
        a_s = self._afterstate_dynamics(s, a)
        sig, q = self._afterstate_predictions(a_s)
        s, r = self._dynamics(a_s, c)

    def call(self, board, act, vec):
        x = self._representation(board, vec)
        return self._predictions(x)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 10, 10, 2), dtype=tf.float32, name='board'),
        tf.TensorSpec(shape=(None, 178), dtype=tf.float32, name='vec')
    ])
    def _representation(self, board, vec):
        x = self.preprocess_obs(board, vec)
        return self.h(x)

    def representation(self, board, vec):
        """Representation function maps from observation to latent state."""
        return self._representation(board[None], vec[None])[0].numpy()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32, name='state')
    ])
    def _predictions(self, state):
        p, v = self.f(state)
        return p, v

    def predictions(self, state):
        """Returns the network predictions for a latent state."""
        p, v = self._predictions(state[None])
        return NetworkOutput(v[0, 0].numpy(), open_policy_map(p[0].numpy()), 0)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32, name='state'),
        tf.TensorSpec(shape=(None, 100), dtype=tf.float32, name='action')
    ])
    def _afterstate_dynamics(self, state, action):
        """Implements the dynamics from latent state and action to afterstate
        ."""
        x = tf.concat([state, action], axis=1)
        return self.phi(x)

    def afterstate_dynamics(self, state, action):
        """Implements the dynamics from latent state and action to afterstate
        ."""
        action = Action(action[0], action[1]).encode().reshape(-1)
        return self._afterstate_dynamics(state[None], action[None])[0].numpy()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32, name='state')
    ])
    def _afterstate_predictions(self, state):
        sig, q = self.psi(state)
        sig = tf.keras.layers.Softmax()(sig)
        return sig, q

    def afterstate_predictions(self, state):
        """Returns the network predictions for an afterstate."""
        # No reward for afterstate transitions.
        sig, q = self._afterstate_predictions(state[None])
        sig = sig[0].numpy()
        policy = {i: p for (i, p) in enumerate(sig)}
        return NetworkOutput(q[0, 0].numpy(), policy)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32, name='state'),
        tf.TensorSpec(shape=(None, 32), dtype=tf.float32, name='chance_outcome')
    ])
    def _dynamics(self, state, chance_outcome):
        x = tf.concat([state, chance_outcome], axis=1)
        s, r = self.g(x)
        return s, r

    def dynamics(self, state, chance_outcome):
        """Implements the dynamics from afterstate and chance outcome to
        state."""
        c = np.zeros((1, self.num_codes), 'float32')
        c[:, chance_outcome] = 1
        s, r = self._dynamics(state[None], c)
        return s[0].numpy(), r[0, 0].numpy()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 10, 10, 2), dtype=tf.float32, name='board'),
        tf.TensorSpec(shape=(None, 178), dtype=tf.float32, name='vec')
    ])
    def _encoder(self, board, vec):
        return self.e(self.preprocess_obs(board, vec))

    def encoder(self, board, vec):
        """An encoder maps an observation to an outcome."""
        return self._encoder(board[None], vec[None])[0].numpy()
