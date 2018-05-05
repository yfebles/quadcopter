import numpy as np

from agents.helpers import EpisodeLearnerStore, NoiseGenerator
from agents.models.fully_connected_actor_model import FullyConnectedActorModel
from agents.models.double_fully_conncted_layers_critic_model import DoubleFullyConnctedlayersCriticModel


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, actor_learning_rate=0.005, critic_learning_rate=0.001, tau=0.01, gamma=0.9,
                 buffer_size=50000, batch_size=64, exploration_mu=0.05, exploration_theta=0.1, exploration_sigma=0.1):

        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = FullyConnectedActorModel(self.state_size, self.action_size, self.action_low, self.action_high, actor_learning_rate)
        self.actor_target = FullyConnectedActorModel(self.state_size, self.action_size, self.action_low, self.action_high, actor_learning_rate)

        # Critic (Value) Model
        self.critic_local = DoubleFullyConnctedlayersCriticModel(self.state_size, self.action_size, critic_learning_rate)
        self.critic_target = DoubleFullyConnctedlayersCriticModel(self.state_size, self.action_size, critic_learning_rate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = NoiseGenerator(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = EpisodeLearnerStore(self.buffer_size, self.batch_size)
        self.score = 0
        self.best_score = 0
        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, reward, done, action, next_state):
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size or done:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        noise = self.noise.sample()
        return list(action + noise)  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        self.score = rewards.max()
        self.best_score = max(self.best_score, self.score)

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        self.update(self.critic_local.model, self.critic_target.model)
        self.update(self.actor_local.model, self.actor_target.model)

    def update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

