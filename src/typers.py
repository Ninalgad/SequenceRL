from typing import (Any, Dict, Callable, List, NamedTuple, Tuple, Union,
                    Optional, Sequence)
import numpy as np


class Color:
    BLUE = 1
    RED = 2

    def get_players():
        return [Color.BLUE, Color.RED]


class Action:
    def __init__(self, x: int , y: int, card: str = ""):
        self.x = x
        self.y = y
        self.card = card

    def __str__(self):
        return f"Action({self.x}, {self.y}, card={self.card})"

    def encode(self):
        a = np.zeros((10, 10), "float32")
        a[self.x, self.y] = 1
        return a


class State(NamedTuple):
    """Data for a single state."""
    observation: Dict[str, float]
    reward: float
    player: Color
    action: Tuple[int]


# List of game states
Trajectory = Sequence[State]
