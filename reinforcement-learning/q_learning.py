"""
Q-Learning Algorithms

This module implements various Q-learning algorithms for reinforcement learning:
- Tabular Q-learning
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN

Author: ML Framework Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from collections import defaultdict
import warnings


class TabularQLearning:
    """
    Tabular Q-Learning Algorithm

    Q-learning is a model-free reinforcement learning algorithm that learns
    the value of an action in a particular state. It uses a Q-table to store
    Q-values for each state-action pair.

    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

    Parameters:
    -----------
    n_states : int
        Number of states in the environment.
    n_actions : int
        Number of actions available.
    learning_rate : float, default=0.1
        Learning rate (alpha).
    discount_factor : float, default=0.99
        Discount factor (gamma) for future rewards.
    epsilon : float, default=0.1
        Exploration rate for epsilon-greedy policy.
    epsilon_decay : float, default=0.995
        Decay rate for epsilon after each episode.
    epsilon_min : float, default=0.01
        Minimum epsilon value.

    Example:
    --------
    >>> q_learning = TabularQLearning(
    ...     n_states=16,
    ...     n_actions=4,
    ...     learning_rate=0.1,
    ...     discount_factor=0.99
    ... )
    >>>
    >>> # Training loop
    >>> for episode in range(1000):
    ...     state = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = q_learning.get_action(state)
    ...         next_state, reward, done, _ = env.step(action)
    ...         q_learning.update(state, action, reward, next_state, done)
    ...         state = next_state
    ...     q_learning.decay_epsilon()
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters:
        -----------
        state : int
            Current state.
        eval_mode : bool, default=False
            If True, always select greedy action (no exploration).

        Returns:
        --------
        action : int
            Selected action.
        """
        if not eval_mode and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: greedy action
            return np.argmax(self.q_table[state])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-value using Q-learning update rule.

        Parameters:
        -----------
        state : int
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int
            Next state.
        done : bool
            Whether episode is done.
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy policy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)


class QNetwork(nn.Module):
    """
    Q-Network for Deep Q-Learning

    Neural network that approximates Q-values.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [64, 64]):
        super(QNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass to get Q-values for all actions."""
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network

    Separates value and advantage functions for better learning.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [64, 64]):
        super(DuelingQNetwork, self).__init__()

        # Shared feature layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_layer = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Linear(hidden_dims[-1], 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        """Forward pass using dueling architecture."""
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DQN:
    """
    Deep Q-Network (DQN)

    Uses a neural network to approximate Q-values. Includes:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration

    Parameters:
    -----------
    state_dim : int
        Dimension of state space.
    action_dim : int
        Number of actions.
    hidden_dims : list, default=[64, 64]
        Hidden layer dimensions.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    discount_factor : float, default=0.99
        Discount factor (gamma).
    epsilon : float, default=1.0
        Initial exploration rate.
    epsilon_decay : float, default=0.995
        Epsilon decay rate.
    epsilon_min : float, default=0.01
        Minimum epsilon.
    batch_size : int, default=64
        Batch size for training.
    target_update_freq : int, default=10
        Frequency to update target network.
    device : str, default='cpu'
        Device to use ('cpu' or 'cuda').

    Example:
    --------
    >>> from replay_buffer import ReplayBuffer
    >>>
    >>> dqn = DQN(state_dim=4, action_dim=2)
    >>> buffer = ReplayBuffer(capacity=10000)
    >>>
    >>> for episode in range(1000):
    ...     state = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = dqn.get_action(state)
    ...         next_state, reward, done, _ = env.step(action)
    ...         buffer.push(state, action, reward, next_state, done)
    ...
    ...         if len(buffer) > batch_size:
    ...             loss = dqn.train_step(buffer, batch_size)
    ...
    ...         state = next_state
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.update_count = 0

    def get_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax().item()

    def train_step(self, replay_buffer, batch_size: Optional[int] = None) -> float:
        """
        Perform one training step.

        Parameters:
        -----------
        replay_buffer : ReplayBuffer
            Experience replay buffer.
        batch_size : int, optional
            Batch size (uses self.batch_size if None).

        Returns:
        --------
        loss : float
            Training loss.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class DoubleDQN(DQN):
    """
    Double DQN

    Reduces overestimation bias by using the online network to select
    actions and the target network to evaluate them.

    Example:
    --------
    >>> ddqn = DoubleDQN(state_dim=4, action_dim=2)
    >>> # Use same interface as DQN
    """

    def train_step(self, replay_buffer, batch_size: Optional[int] = None) -> float:
        """Training step with Double DQN update."""
        if batch_size is None:
            batch_size = self.batch_size

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


class DuelingDQN(DQN):
    """
    Dueling DQN

    Uses dueling architecture that separates value and advantage functions.

    Example:
    --------
    >>> dueling_dqn = DuelingDQN(state_dim=4, action_dim=2)
    >>> # Use same interface as DQN
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = 'cpu'
    ):
        # Initialize parent but override networks
        super().__init__(
            state_dim, action_dim, hidden_dims, learning_rate,
            discount_factor, epsilon, epsilon_decay, epsilon_min,
            batch_size, target_update_freq, device
        )

        # Replace with dueling networks
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)


if __name__ == "__main__":
    print("=" * 70)
    print("Q-LEARNING EXAMPLES")
    print("=" * 70)

    # Example 1: Tabular Q-learning
    print("\n1. Tabular Q-Learning")
    print("-" * 70)

    q_learning = TabularQLearning(
        n_states=16,
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.99
    )

    print(f"Q-table shape: {q_learning.q_table.shape}")
    print(f"Initial epsilon: {q_learning.epsilon}")

    # Simulate some updates
    for i in range(10):
        state = np.random.randint(16)
        action = q_learning.get_action(state)
        next_state = np.random.randint(16)
        reward = np.random.randn()
        done = np.random.random() < 0.1

        q_learning.update(state, action, reward, next_state, done)

    print(f"Q-values for state 0: {q_learning.q_table[0]}")

    # Example 2: DQN
    print("\n2. Deep Q-Network (DQN)")
    print("-" * 70)

    dqn = DQN(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        learning_rate=0.001
    )

    print(f"Q-network: {dqn.q_network}")
    print(f"Number of parameters: {sum(p.numel() for p in dqn.q_network.parameters())}")

    # Test forward pass
    test_state = np.random.randn(4)
    action = dqn.get_action(test_state)
    print(f"Selected action: {action}")

    # Example 3: Double DQN
    print("\n3. Double DQN")
    print("-" * 70)

    ddqn = DoubleDQN(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64]
    )

    print("Double DQN initialized (reduces overestimation bias)")

    # Example 4: Dueling DQN
    print("\n4. Dueling DQN")
    print("-" * 70)

    dueling_dqn = DuelingDQN(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64]
    )

    print(f"Dueling architecture: {dueling_dqn.q_network}")

    print("\n" + "=" * 70)
    print("All Q-learning examples completed!")
    print("=" * 70)
