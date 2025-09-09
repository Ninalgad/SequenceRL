import numpy as np
from env import SequenceGameEnv
from typing import List
from actor import Actor
from typers import Action


class CenterActor(Actor):

    def reset(self):
        return

    def select_action(self, env: SequenceGameEnv, legal_actions: List[Action]):
        dists = [abs(5 - a.x) + abs(5 - a.y) for a in legal_actions]
        return legal_actions[np.argmin(dists)]
