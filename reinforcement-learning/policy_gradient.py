"""
Policy Gradient Methods

This module implements policy gradient algorithms:
- REINFORCE (Monte Carlo Policy Gradient)
- PPO (Proximal Policy Optimization)
- TRPO concepts

Author: ML Framework Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional
import warnings


class PolicyNetwork(nn.Module):
    """
    Policy Network for discrete actions.

    Outputs probability distribution over actions.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [64, 64]):
        super(PolicyNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        """Forward pass to get action probabilities."""
        features = self.network(state)
        action_probs = F.softmax(self.action_head(features), dim=-1)
        return action_probs


class ContinuousPolicyNetwork(nn.Module):
    """
    Policy Network for continuous actions.

    Outputs mean and log standard deviation for Gaussian policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [64, 64]):
        super(ContinuousPolicyNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        """Forward pass to get action distribution parameters."""
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilize
        return mean, log_std


class REINFORCE:
    """
    REINFORCE Algorithm (Monte Carlo Policy Gradient)

    Basic policy gradient algorithm that updates policy using
    complete episode returns.

    Update rule:
    ∇θ J(θ) ≈ E[∇θ log π(a|s) * G_t]

    where G_t is the return from timestep t.

    Parameters:
    -----------
    state_dim : int
        Dimension of state space.
    action_dim : int
        Number of actions (discrete) or dimension of action space (continuous).
    hidden_dims : list, default=[64, 64]
        Hidden layer dimensions.
    learning_rate : float, default=0.001
        Learning rate for policy optimizer.
    discount_factor : float, default=0.99
        Discount factor (gamma).
    continuous : bool, default=False
        Whether action space is continuous.
    device : str, default='cpu'
        Device to use.

    Example:
    --------
    >>> reinforce = REINFORCE(state_dim=4, action_dim=2)
    >>>
    >>> for episode in range(1000):
    ...     states, actions, rewards = [], [], []
    ...     state = env.reset()
    ...     done = False
    ...
    ...     while not done:
    ...         action, log_prob = reinforce.get_action(state)
    ...         next_state, reward, done, _ = env.step(action)
    ...
    ...         states.append(state)
    ...         actions.append(action)
    ...         rewards.append(reward)
    ...
    ...         state = next_state
    ...
    ...     # Update policy after episode
    ...     loss = reinforce.update(states, actions, rewards)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        continuous: bool = False,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.continuous = continuous
        self.device = torch.device(device)

        # Create policy network
        if continuous:
            self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Storage for episode
        self.saved_log_probs = []
        self.rewards = []

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select action from policy.

        Parameters:
        -----------
        state : np.ndarray
            Current state.
        deterministic : bool, default=False
            If True, select most likely action (no sampling).

        Returns:
        --------
        action : int or np.ndarray
            Selected action.
        log_prob : float
            Log probability of the action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.continuous:
            mean, log_std = self.policy(state_tensor)
            std = torch.exp(log_std)

            if deterministic:
                action = mean
                log_prob = torch.zeros(1)
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

            return action.detach().cpu().numpy()[0], log_prob.item()

        else:
            action_probs = self.policy(state_tensor)
            dist = Categorical(action_probs)

            if deterministic:
                action = action_probs.argmax()
                log_prob = torch.zeros(1)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item()

    def update(
        self,
        states: List[np.ndarray],
        actions: List,
        rewards: List[float]
    ) -> float:
        """
        Update policy using REINFORCE algorithm.

        Parameters:
        -----------
        states : list
            States from episode.
        actions : list
            Actions from episode.
        rewards : list
            Rewards from episode.

        Returns:
        --------
        loss : float
            Policy loss.
        """
        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns (helps training stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_loss = []

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)

        if self.continuous:
            actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
            mean, log_std = self.policy(states_tensor)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions_tensor).sum(dim=1)
        else:
            actions_tensor = torch.LongTensor(actions).to(self.device)
            action_probs = self.policy(states_tensor)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions_tensor)

        # REINFORCE objective: maximize E[log π(a|s) * G]
        # Minimize negative objective
        loss = -(log_probs * returns).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, filepath: str):
        """Save policy."""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load policy."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class PPO:
    """
    Proximal Policy Optimization (PPO)

    State-of-the-art policy gradient method that uses a clipped
    surrogate objective to prevent too large policy updates.

    Objective:
    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

    Parameters:
    -----------
    state_dim : int
        Dimension of state space.
    action_dim : int
        Number of actions.
    hidden_dims : list, default=[64, 64]
        Hidden layer dimensions.
    learning_rate : float, default=0.0003
        Learning rate.
    discount_factor : float, default=0.99
        Discount factor (gamma).
    gae_lambda : float, default=0.95
        GAE lambda for advantage estimation.
    clip_epsilon : float, default=0.2
        PPO clipping parameter.
    value_coef : float, default=0.5
        Value loss coefficient.
    entropy_coef : float, default=0.01
        Entropy bonus coefficient.
    epochs : int, default=10
        Number of epochs per update.
    continuous : bool, default=False
        Whether action space is continuous.
    device : str, default='cpu'
        Device to use.

    Example:
    --------
    >>> ppo = PPO(state_dim=4, action_dim=2)
    >>>
    >>> # Collect trajectories
    >>> states, actions, rewards, next_states, dones = collect_trajectories(env, ppo)
    >>>
    >>> # Update policy
    >>> loss = ppo.update(states, actions, rewards, next_states, dones)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 10,
        continuous: bool = False,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.continuous = continuous
        self.device = torch.device(device)

        # Policy network
        if continuous:
            self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        # Value network (critic)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate
        )

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.continuous:
                mean, log_std = self.policy(state_tensor)
                std = torch.exp(log_std)

                if deterministic:
                    action = mean
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()

                return action.cpu().numpy()[0]
            else:
                action_probs = self.policy(state_tensor)
                dist = Categorical(action_probs)

                if deterministic:
                    action = action_probs.argmax()
                else:
                    action = dist.sample()

                return action.item()

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Parameters:
        -----------
        rewards : np.ndarray
            Rewards.
        values : np.ndarray
            Value estimates for states.
        next_values : np.ndarray
            Value estimates for next states.
        dones : np.ndarray
            Done flags.

        Returns:
        --------
        advantages : np.ndarray
            Advantage estimates.
        returns : np.ndarray
            Return estimates.
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Returns:
        --------
        metrics : dict
            Training metrics.
        """
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get values
        with torch.no_grad():
            values = self.value_network(states).squeeze().cpu().numpy()
            next_values = self.value_network(next_states).squeeze().cpu().numpy()

        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            rewards.cpu().numpy(),
            values,
            next_values,
            dones.cpu().numpy()
        )

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old log probs
        with torch.no_grad():
            if self.continuous:
                actions_tensor = torch.FloatTensor(actions).to(self.device)
                mean, log_std = self.policy(states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                old_log_probs = dist.log_prob(actions_tensor).sum(dim=1)
            else:
                actions_tensor = torch.LongTensor(actions).to(self.device)
                action_probs = self.policy(states)
                dist = Categorical(action_probs)
                old_log_probs = dist.log_prob(actions_tensor)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            # Get current log probs and entropy
            if self.continuous:
                mean, log_std = self.policy(states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions_tensor).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()
            else:
                action_probs = self.policy(states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

            # Ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred = self.value_network(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            'policy_loss': total_policy_loss / self.epochs,
            'value_loss': total_value_loss / self.epochs,
            'entropy': total_entropy / self.epochs
        }

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_network.load_state_dict(checkpoint['value'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    print("=" * 70)
    print("POLICY GRADIENT EXAMPLES")
    print("=" * 70)

    # Example 1: REINFORCE
    print("\n1. REINFORCE Algorithm")
    print("-" * 70)

    reinforce = REINFORCE(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        learning_rate=0.001
    )

    print(f"Policy network: {reinforce.policy}")

    # Simulate episode
    states = [np.random.randn(4) for _ in range(10)]
    actions = []
    rewards = []

    for state in states:
        action, log_prob = reinforce.get_action(state)
        actions.append(action)
        rewards.append(np.random.randn())

    loss = reinforce.update(states, actions, rewards)
    print(f"Training loss: {loss:.4f}")

    # Example 2: PPO
    print("\n2. Proximal Policy Optimization (PPO)")
    print("-" * 70)

    ppo = PPO(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        learning_rate=0.0003
    )

    print(f"Policy network: {ppo.policy}")
    print(f"Value network: {ppo.value_network}")

    # Simulate batch
    n_steps = 128
    states = np.random.randn(n_steps, 4)
    actions = np.random.randint(0, 2, n_steps)
    rewards = np.random.randn(n_steps)
    next_states = np.random.randn(n_steps, 4)
    dones = np.zeros(n_steps)

    metrics = ppo.update(states, actions, rewards, next_states, dones)
    print(f"Policy loss: {metrics['policy_loss']:.4f}")
    print(f"Value loss: {metrics['value_loss']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")

    print("\n" + "=" * 70)
    print("All policy gradient examples completed!")
    print("=" * 70)
