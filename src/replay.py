import numpy as np
from scipy.signal import lfilter
from config import DQNConfig
from typers import *


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ReplayBuffer:
    """A replay buffer to hold the experience generated by the selfplay."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.data = []

    def __len__(self):
        return len(self.data)

    def finish_trajectory(self, seq: Trajectory, last_rewards: dict):
        # get rewards by color
        rewards = {c: [] for c in Color.get_players()}
        for s in seq:
            rewards[s.player].append(s.reward)

        # left shift rewards and append last reward
        rewards = {c: r[1:] + [last_rewards[c]] for c, r in rewards.items()}

        # smooth out rewards to reduce training variance
        for c, r in rewards.items():
            rewards[c] = list(discounted_cumulative_sums(rewards[c], self.config.discount))

        # reassign rewards
        for i, s in enumerate(seq):
            seq[i] = s._replace(reward=rewards[s.player].pop(0))

        return seq

    def save(self, seq: Trajectory, last_rewards: dict):
        seq = self.finish_trajectory(seq, last_rewards)
        if len(self.data) > self.config.num_trajectories_in_buffer:
            # Remove the oldest sequence from the buffer.
            self.data.pop(0)
        self.data.append(seq)

    def sample_trajectory(self) -> Trajectory:
        """Samples a trajectory uniformly or using prioritization."""
        return self.data[np.random.choice(len(self))]

    def sample_element(self) -> Trajectory:
        """Samples a single element from the buffer."""
        # Sample a trajectory.
        trajectory = self.sample_trajectory()
        state_idx = np.random.choice(len(trajectory))

        # Returns a trajectory of experiment.
        return trajectory[state_idx]

    def sample(self) -> Sequence[Trajectory]:
        """Samples a training batch."""
        return [self.sample_element() for _ in range(self.config.batch_size)]