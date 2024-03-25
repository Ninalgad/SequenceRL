import numpy as np
import sys; sys.path.append("..")
from env import SequenceGameEnv
from typing import List
from actor import Actor
from typers import Action


class RandomActor(Actor):

    def reset(self):
        return

    def select_action(self, env: SequenceGameEnv, legal_actions: List[Action]):
        idx = np.random.choice(len(legal_actions))
        return legal_actions[idx]
