import random
import numpy as np
from collections import deque, namedtuple


class EpisodeLearnerStore:
    def __init__(self, buffer_size, batch_size):
        self.episodes = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.episodes.append(e)

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""

        batch_size = self.batch_size if batch_size is None else batch_size
        return random.sample(self.episodes, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.episodes)


class NoiseGenerator:
    """
    Simple noise random genrator multidim.
    Parametrized by dimensions mu, th and sigma for an state size defined.
    """
    def __init__(self, size, mu, th, sig):
        self.th = th
        self.sig = sig
        self.mu = mu * np.ones(size)
        self.state = self.mu

    def reset(self):
        self.state = self.mu

    def sample(self):
        old_state = self.state
        exploration_ratio = np.random.randn(len(old_state))
        self.state = old_state + (self.th * (self.mu - old_state) + self.sig * exploration_ratio)
        return self.state
