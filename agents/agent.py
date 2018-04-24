import numpy as np


class PolicyDrivenAgent:
    def __init__(self, task):
        # Task (environment) information

        self.task = task
        self.best_w = None
        self.noise_scale = 0.1
        self.reward_decay = .9
        self.best_score = -np.inf

        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.count, self.score, self.total_reward = 0, 0, 0.0
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (4 * self.state_size))) # start producing actions in a decent range

        self.reset_episode()

    def reset_episode(self):
        self.count, self.total_reward = 0, 0.0
        return self.task.reset()

    def step(self, reward, done):
        self.count += 1
        self.total_reward = self.total_reward * self.reward_decay + reward

        if done: # Learn, if at end of episode
            self.learn()

    def act(self, state):  # Choose action based on given state and policy
        return np.dot(state, self.w)  # simple linear policy action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions


class ActorCriticAgent:
    pass
