from typing import (Any, Dict, Callable, List, NamedTuple, Tuple, Union,
                    Optional, Sequence)


class Color:
    BLUE = 1
    RED = 2

    def get_players():
        return [Color.BLUE, Color.RED]


class Action(NamedTuple):
    x: int
    y: int
    card: str


class State(NamedTuple):
    """Data for a single state."""
    observation: Dict[str, float]
    reward: float
    player: Color


# List of game states
Trajectory = Sequence[State]
