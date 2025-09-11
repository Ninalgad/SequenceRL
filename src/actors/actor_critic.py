import numpy as np
import sys; sys.path.append("..")
from actor import Actor
from algorithm import Algorithm
from scipy.special import softmax


class A2CActor(Actor):
    def __init__(self, algo: Algorithm, training: bool = False, temperature: float = 0.1,
                 verbose: bool = False):
        super(A2CActor, self).__init__()
        self.algo = algo
        self.training = training
        self.temp = temperature
        self.num_actions = 0
        self.verbose = verbose

    def reset(self):
        self.num_actions = 0

    def select_action(self, env, actions):
        self.num_actions += 1
        if self.training:
          if np.random.uniform() < .01:
            return actions[np.random.choice(len(actions))]
        obs = env.observation()
        qval, logits = self.algo.policy(obs['board'], obs['vec'])
        action_probs = softmax([logits[a.x, a.y] / self.temp for a in actions])

        action_idx = np.random.choice(len(actions), p=action_probs)
        action = actions[action_idx]

        if self.verbose:
            print(f'Eval: {qval:.4f}')

        return action
