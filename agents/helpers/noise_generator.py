import numpy as np


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

