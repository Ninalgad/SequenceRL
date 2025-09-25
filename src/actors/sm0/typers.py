from typing import Any, Dict, Callable, List, NamedTuple, Tuple, Union, Optional, Sequence
from typers import *
import numpy as np


MAXIMUM_FLOAT_VALUE = float('inf')
ActionOrOutcome = Any
Player = Color


class NetworkOutput(NamedTuple):
    value: float
    probabilities: np.array
    reward: Optional[float] = 0.0


class KnownBounds(NamedTuple):
    min: float
    max: float
