import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
  """Interacts with and learns from the environment."""

  def __init__(self, config):
    """Initialize an Agent object.

    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        random_seed (int): random seed
    """
    self.state_size = config.state_size
    self.action_size = config.action_size
    self.seed = random.seed(config.random_seed)
    self.config = config
    self.t_step = 0
    # Actor Network (w/ Target Network)
    self.actor_local = Actor(
        self.state_size,
        self.action_size,
        config.random_seed
    ).to(device)
    self.actor_target = Actor(
        self.state_size,
        self.action_size,
        config.random_seed
    ).to(device)
    self.actor_optimizer = optim.Adam(
        self.actor_local.parameters(),
        lr=config.lr_actor
    )

    # Critic Network (w/ Target Network)
    self.critic_local = Critic(
        self.state_size,
        self.action_size,
        config.random_seed
    ).to(device)
    self.critic_target = Critic(
        self.state_size,
        self.action_size,
        config.random_seed
    ).to(device)
    self.critic_optimizer = optim.Adam(
        self.critic_local.parameters(),
        lr=config.lr_critic,
        weight_decay=config.weight_decay
    )

    # Noise process
    self.noise = OUNoise(self.action_size, config.random_seed)

    # Replay memory
    self.memory = ReplayBuffer(
        self.action_size, config.buffer_size, config.batch_size, config.random_seed)

  def step(self, states, actions, rewards, next_states, dones):
    """Save experience in replay memory, and use random sample from buffer to learn."""
    # Save experience / reward
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
      self.memory.add(state, action, reward, next_state, done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.config.update_every
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > self.config.batch_size:
        experiences = self.memory.sample()
        self.learn(experiences, self.config.gamma)

  def act(self, state, add_noise=True):
    """Returns actions for given state as per current policy."""
    state = torch.from_numpy(state).float().to(device)
    self.actor_local.eval()
    with torch.no_grad():
      action = self.actor_local(state).cpu().data.numpy()
    self.actor_local.train()
    if add_noise:
      action += self.noise.sample()
    return np.clip(action, -1, 1)

  def reset(self):
    self.noise.reset()

  def learn(self, experiences, gamma):
    """Update policy and value parameters using given batch of experience tuples.
    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    where:
        actor_target(state) -> action
        critic_target(state, action) -> Q-value

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    # ---------------------------- update critic ---------------------------- #
    # Get predicted next-state actions and Q values from target models
    actions_next = self.actor_target(next_states)
    Q_targets_next = self.critic_target(next_states, actions_next)
    # Compute Q targets for current states (y_i)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    # Compute critic loss
    Q_expected = self.critic_local(states, actions)
    critic_loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
    self.critic_optimizer.step()

    # ---------------------------- update actor ---------------------------- #
    # Compute actor loss
    actions_pred = self.actor_local(states)
    actor_loss = -self.critic_local(states, actions_pred).mean()
    # Minimize the loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
    self.actor_optimizer.step()

    # ----------------------- update target networks ----------------------- #
    self.soft_update(self.critic_local, self.critic_target, self.config.tau)
    self.soft_update(self.actor_local, self.actor_target, self.config.tau)

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(
          tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
  """Ornstein-Uhlenbeck process."""

  def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
    """Initialize parameters and noise process."""
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.seed = random.seed(seed)
    self.reset()

  def reset(self):
    """Reset the internal state (= noise) to mean (mu)."""
    self.state = copy.copy(self.mu)

  def sample(self):
    """Update internal state and return it as a noise sample."""
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * \
        np.array([random.random() for i in range(len(x))])
    self.state = x + dx
    return self.state
