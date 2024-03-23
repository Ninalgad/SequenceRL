import numpy as np
from abc import ABCMeta, abstractmethod
from typers import *
from config import DQNConfig
from algorithm import Algorithm
from replay import ReplayBuffer


class Learner(metaclass=ABCMeta):
    """An learner to update the network weights based."""

    @abstractmethod
    def learn(self):
        """Single training step of the learner."""


class DQNLearner(Learner):
    """Implements the learning for Stochastic MuZero."""

    def __init__(self, algo: Algorithm,
                 config: DQNConfig,
                 replay_buffer: ReplayBuffer):
        self.config = config
        self.replay_buffer = replay_buffer
        self.algo = algo

    def get_loss(self):
        return self.algo.train_loss.result().numpy()

    def reset_loss(self):
        self.algo.train_loss.reset_states()

    def sample_batch(self):
        traj_batch = self.replay_buffer.sample()
        training_batch = {'board': [], 'vec': [], 'tar': []}

        for state in traj_batch:
            training_batch['board'].append(state.observation['board'])
            training_batch['vec'].append(state.observation['vec'])
            training_batch['tar'].append(state.reward)

        training_batch = {k: np.array(v, 'float32') for (k, v) in training_batch.items()}
        training_batch['tar'] = np.expand_dims(training_batch['tar'], -1)

        return training_batch

    def learn(self):
        """Applies a single training step."""

        batch = self.sample_batch()
        loss = self.algo.train_step(**batch).numpy()
        return loss

    def export(self) -> Algorithm:
        return self.algo
