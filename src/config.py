import dataclasses
from typers import *
from env import SequenceGameEnv


# Returns an instance of the environment.
EnvironmentFactory = Callable[[], SequenceGameEnv]


@dataclasses.dataclass
class DQNConfig:
    # A factory for the environment.
    environment_factory: EnvironmentFactory

    # Self-Play
    discount: float

    # Replay buffer.
    num_trajectories_in_buffer: int = 1_000
    batch_size: int = 64

    # Training
    training_steps: int = int(1e6)
    export_network_every: int = int(250)
    learning_rate: float = 1e-5
    training_steps_per_epoch: int = 30
    games_per_epoch: int = 20


def sequence_1v1_config() -> DQNConfig:
    """Returns the config for the game of Sequence."""

    def environment_factory():
        return SequenceGameEnv()

    return DQNConfig(
        environment_factory=environment_factory,
        discount=0.99,
        num_trajectories_in_buffer=int(1e5),
        training_steps=int(8e6),
        batch_size=64,
        learning_rate=1e-5)
