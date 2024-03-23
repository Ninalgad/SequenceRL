from abc import ABCMeta, abstractmethod
from typers import Color
from env import SequenceGameEnv
from typing import List
from typers import Action


class Actor(metaclass=ABCMeta):
    """An actor to interact with the environment."""

    def __init__(self, color: Color = None) -> None:
        self.color = color

    @abstractmethod
    def reset(self):
        """Resets the player for a new episode."""

    @abstractmethod
    def select_action(self, env: SequenceGameEnv, legal_actions: List[Action]) -> Action:
        """Selects an action for the current state of the environment."""

    def set_color(self, color: Color):
        self.color = color
