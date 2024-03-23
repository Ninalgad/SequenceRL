import numpy as np
from copy import deepcopy
from src.actor import Actor
from src.algorithm import Algorithm


def collate_states(states):
    x = states.pop(0)
    x = {k: [v] for (k, v) in x.items()}
    for s in states:
        for k, v in s.items():
            x[k].append(v)

    return x


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

        best_idx = None
        next_states = []
        for i, a in enumerate(actions):
            env_copy = deepcopy(env)
            env_copy.apply(a)

            if env.winner() == self.color:
                best_idx = i
                break

            next_states.append(env_copy.observation(reverse=True))

        if best_idx is None:
            next_states = collate_states(next_states)
            q = self.algo.policy(next_states['board'], next_states['vec'])
            q = np.squeeze(q.numpy(), -1)
            best_idx = np.argmax(q)

            if self.verbose:
                print(f'Eval: {q[best_idx]:.4f}')

        return actions[best_idx]
