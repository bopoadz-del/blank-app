"""
Actor-Critic Methods

This module implements Actor-Critic algorithms:
- A2C (Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)

Author: ML Framework Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import copy


class ActorNetwork(nn.Module):
    """Actor network for continuous control."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 256], max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic network (Q-function)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 256]):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class A2C:
    """
    Advantage Actor-Critic (A2C)

    Synchronous version of A3C. Uses advantage function to reduce variance.

    Parameters:
    -----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    hidden_dims : list, default=[64, 64]
        Hidden layer dimensions.
    learning_rate : float, default=0.001
        Learning rate.
    discount_factor : float, default=0.99
        Discount factor.
    value_coef : float, default=0.5
        Value loss coefficient.
    entropy_coef : float, default=0.01
        Entropy coefficient.
    device : str, default='cpu'
        Device.

    Example:
    --------
    >>> a2c = A2C(state_dim=4, action_dim=2)
    >>> # Collect trajectories and update
    >>> loss = a2c.update(states, actions, rewards, next_states, dones)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=[64, 64],
        learning_rate=0.001,
        discount_factor=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        device='cpu'
    ):
        self.discount_factor = discount_factor
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

    def get_action(self, state: np.ndarray):
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        """Update actor-critic."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        # Compute advantages
        targets = rewards + (1 - dones) * self.discount_factor * next_values
        advantages = (targets - values).detach()

        # Actor loss
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, targets.detach())

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG)

    Off-policy actor-critic for continuous control.

    Parameters:
    -----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    max_action : float, default=1.0
        Maximum action value.
    hidden_dims : list, default=[256, 256]
        Hidden dimensions.
    learning_rate : float, default=0.001
        Learning rate.
    discount_factor : float, default=0.99
        Discount factor.
    tau : float, default=0.005
        Soft update parameter.
    device : str, default='cpu'
        Device.

    Example:
    --------
    >>> ddpg = DDPG(state_dim=4, action_dim=2, max_action=2.0)
    >>> action = ddpg.get_action(state)
    >>> loss = ddpg.update(replay_buffer, batch_size=64)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims=[256, 256],
        learning_rate=0.001,
        discount_factor=0.99,
        tau=0.005,
        device='cpu'
    ):
        self.discount_factor = discount_factor
        self.tau = tau
        self.max_action = max_action
        self.device = torch.device(device)

        # Actor
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Critic
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_action(self, state: np.ndarray, noise=0.1):
        """Select action with optional noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer, batch_size=64):
        """Update DDPG."""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.discount_factor * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def _soft_update(self, source, target):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":
    print("=" * 70)
    print("ACTOR-CRITIC EXAMPLES")
    print("=" * 70)

    # Example 1: A2C
    print("\n1. A2C (Advantage Actor-Critic)")
    print("-" * 70)

    a2c = A2C(state_dim=4, action_dim=2)
    print(f"Actor: {a2c.actor}")
    print(f"Critic: {a2c.critic}")

    # Simulate batch
    states = np.random.randn(32, 4)
    actions = np.random.randint(0, 2, 32)
    rewards = np.random.randn(32)
    next_states = np.random.randn(32, 4)
    dones = np.zeros(32)

    metrics = a2c.update(states, actions, rewards, next_states, dones)
    print(f"Actor loss: {metrics['actor_loss']:.4f}")
    print(f"Critic loss: {metrics['critic_loss']:.4f}")

    # Example 2: DDPG
    print("\n2. DDPG (Deep Deterministic Policy Gradient)")
    print("-" * 70)

    ddpg = DDPG(state_dim=4, action_dim=2, max_action=2.0)
    print(f"Actor: {ddpg.actor}")

    # Test action selection
    state = np.random.randn(4)
    action = ddpg.get_action(state, noise=0.1)
    print(f"Action: {action}")

    print("\n" + "=" * 70)
    print("All actor-critic examples completed!")
    print("=" * 70)
