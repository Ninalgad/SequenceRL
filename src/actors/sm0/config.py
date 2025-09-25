import dataclasses
from .typers import *


@dataclasses.dataclass
class StochasticMuZeroConfig:
    # Self-Play
    num_actors: int
    num_simulations: int
    discount: float

    # Root prior exploration noise.
    root_dirichlet_alpha: float
    root_dirichlet_fraction: float
    root_dirichlet_adaptive: bool

    # UCB formula
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    known_bounds: Optional[KnownBounds] = None

    visit_softmax_temperature_fn: Callable = lambda step: 1

    # Replay buffer.
    num_unroll_steps: int = 5
    td_steps: int = 6
    td_lambda: float = 1.0
    # Alpha and beta parameters for prioritization.
    # By default they are set to 0 which means uniform sampling.
    priority_alpha: float = 0.0
    priority_beta: float = 0.0

    # Reservoir sampling to be used only for NFSP.
    revervoir_replay_size: int = -1

    # A factor to decide the ratio between the average policy acting
    # and the best response one (Stochastic MuZero).
    # See https://arxiv.org/abs/1603.01121 for more details.
    anticipatory_factor: float = 0.1

    # Training
    training_steps: int = int(1e6)
    export_network_every: int = int(300)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_trajectories_in_buffer: int = 500

    # The number of chance codes (codebook size).
    # We use a codebook of size 32 for all our experiments.
    codebook_size: int = 32

    # Data generation
    games_per_epoch = 2
    training_steps_per_epoch: int = 52
    self_play_games_per_epoch: int = 2

    # League
    max_league_size: int = 10
    run_n_leagueplay_games_per_epoch: int = 2


def sequence_sm0_config() -> StochasticMuZeroConfig:
    def visit_softmax_temperature(train_steps: int) -> float:
        return 1.0

    return StochasticMuZeroConfig(
        num_actors=1000,
        num_simulations=1000,
        discount=1.0,
        # Unused, we use adaptive dirichlet for backgammon.
        root_dirichlet_alpha=-1.0,
        root_dirichlet_fraction=0.1,
        root_dirichlet_adaptive=True,
        known_bounds=KnownBounds(min=-1, max=1),
        # We use monte carlo returns.
        td_steps=int(1e3),
        training_steps=int(8e6),
        batch_size=5,
        learning_rate=3e-4,
        weight_decay=1e-4)
