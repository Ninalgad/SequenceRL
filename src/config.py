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
    num_trajectories_in_buffer: int = 500
    batch_size: int = int(64)

    # Training
    training_steps: int = int(1e6)
    export_network_every: int = 100
    learning_rate: float = 1e-5
    training_steps_per_epoch: int = 3
    games_per_epoch: int = 2


def sequence_1v1_config() -> DQNConfig:
    """Returns the config for the game of Sequence."""

    def environment_factory():
        return SequenceGameEnv()

    return DQNConfig(
        environment_factory=environment_factory,
        discount=0.99,
        num_trajectories_in_buffer=int(1e4),
        training_steps=int(8e6),
        batch_size=64,
        learning_rate=1e-5)


@dataclasses.dataclass
class DQNLeagueConfig:
    # A factory for the environment.
    environment_factory: EnvironmentFactory

    # Self-Play
    discount: float

    # Replay buffer.
    num_trajectories_in_buffer: int = 500
    batch_size: int = int(64)

    # Training
    training_steps: int = int(1e6)
    export_network_every: int = int(300)
    learning_rate: float = 1e-5
    training_steps_per_epoch: int = 4
    self_play_games_per_epoch: int = 2

    # league
    max_league_size: int = 10
    run_n_leagueplay_games_per_epoch: int = 2


def sequence_1v1_league_config() -> DQNLeagueConfig:
    """Returns the config for the game of Sequence."""

    def environment_factory():
        return SequenceGameEnv()

    return DQNLeagueConfig(
        environment_factory=environment_factory,
        discount=0.99,
        num_trajectories_in_buffer=int(1e4),
        training_steps=int(8e6),
        batch_size=64,
        learning_rate=1e-5)
