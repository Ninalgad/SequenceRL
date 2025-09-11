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

        next_states = []
        for i, a in enumerate(actions):
            env_copy = deepcopy(env)
            env_copy.apply(a)

            next_states.append(env_copy.observation(reverse=True))

        next_states = collate_states(next_states)
        q = self.algo.policy(next_states['board'], next_states['vec'])[0]
        q = np.squeeze(q.numpy(), -1)
        best_idx = np.argmax(q)
        action = actions[best_idx]

        if self.verbose:
            print(f'Eval: {q[best_idx]:.4f}')

        return action
