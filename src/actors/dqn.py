import numpy as np
from copy import deepcopy
import sys; sys.path.append("..")
from actor import Actor
from algorithm import Algorithm
from utils import collate_states


class DQNActor(Actor):
    def __init__(self, algo: Algorithm, training: bool = False, verbose: bool = False):
        super(DQNActor, self).__init__()
        self.algo = algo
        self.training = training
        self.num_actions = 0
        self.verbose = verbose

    def reset(self):
        self.num_actions = 0

    def select_action(self, env, actions):
        self.num_actions += 1
        if self.training and (np.random.uniform() < .1):
            return actions[np.random.choice(len(actions))]
        obs = env.observation()

        next_states = []
        for i, a in enumerate(actions):
            obs_copy = deepcopy(obs)
            obs_copy['action'] = a.encode()
            next_states.append(obs_copy)

        next_states = collate_states(next_states)
        q = self.algo.policy(**next_states)
        q = np.squeeze(q.numpy(), -1)
        best_idx = np.argmax(q)
        action = actions[best_idx]

        if self.verbose:
            print(f'Eval| Best: {q[best_idx]:.4f}, Mean: {q.mean():.4f}')

        return action
