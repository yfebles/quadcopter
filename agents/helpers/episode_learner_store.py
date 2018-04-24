import random
from collections import namedtuple, deque


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
