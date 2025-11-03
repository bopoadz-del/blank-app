"""
Reinforcement Learning Utilities

This module provides essential utilities for RL:
- Replay buffers
- Basic environments
- Reward shaping functions

Author: ML Framework Team
"""

import numpy as np
from collections import deque
from typing import Tuple, List
import random


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores transitions and samples random batches for training.

    Parameters:
    -----------
    capacity : int
        Maximum buffer size.

    Example:
    --------
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch.

        Returns:
        --------
        states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer

    Samples important transitions more frequently.

    Parameters:
    -----------
    capacity : int
        Maximum buffer size.
    alpha : float, default=0.6
        How much prioritization to use (0 = uniform).
    beta : float, default=0.4
        Importance sampling correction (increases to 1).
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition with max priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch based on priorities."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get samples
        batch = [self.buffer[idx] for idx in indices]

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# REWARD SHAPING
# ============================================================================

def normalize_reward(reward: float, running_mean: float, running_std: float) -> float:
    """Normalize reward using running statistics."""
    return (reward - running_mean) / (running_std + 1e-8)


def clip_reward(reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clip reward to range."""
    return np.clip(reward, min_val, max_val)


def shaped_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    reward: float,
    gamma: float = 0.99,
    potential_fn = None
) -> float:
    """
    Potential-based reward shaping.

    F(s, a, s') = γΦ(s') - Φ(s)

    where Φ is a potential function.
    """
    if potential_fn is None:
        return reward

    shaped = reward + gamma * potential_fn(next_state) - potential_fn(state)
    return shaped


class RewardNormalizer:
    """
    Running reward normalization.

    Example:
    --------
    >>> normalizer = RewardNormalizer()
    >>> normalized_reward = normalizer.normalize(reward)
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.returns = 0
        self.mean = 0
        self.var = 1
        self.count = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward."""
        self.returns = self.returns * self.gamma + reward
        self.count += 1

        # Update running mean and variance
        delta = self.returns - self.mean
        self.mean += delta / self.count
        self.var += delta * (self.returns - self.mean)

        std = np.sqrt(self.var / self.count) if self.count > 1 else 1.0

        return reward / (std + 1e-8)


# ============================================================================
# SIMPLE ENVIRONMENTS
# ============================================================================

class GridWorld:
    """
    Simple GridWorld Environment

    Agent navigates a grid to reach a goal while avoiding obstacles.

    Parameters:
    -----------
    size : int, default=5
        Grid size (size x size).
    goal : tuple, optional
        Goal position. If None, placed at bottom-right.
    obstacles : list, optional
        List of obstacle positions.

    Example:
    --------
    >>> env = GridWorld(size=5)
    >>> state = env.reset()
    >>> next_state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        size: int = 5,
        goal: Tuple[int, int] = None,
        obstacles: List[Tuple[int, int]] = None
    ):
        self.size = size
        self.goal = goal if goal else (size-1, size-1)
        self.obstacles = obstacles if obstacles else []

        self.action_space = 4  # up, down, left, right
        self.state_space = size * size

        self.agent_pos = None

    def reset(self) -> int:
        """Reset environment and return initial state."""
        # Random start position (not goal or obstacle)
        while True:
            pos = (np.random.randint(self.size), np.random.randint(self.size))
            if pos != self.goal and pos not in self.obstacles:
                self.agent_pos = pos
                break

        return self._get_state()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take action in environment.

        Actions: 0=up, 1=down, 2=left, 3=right

        Returns:
        --------
        state, reward, done, info
        """
        # Move agent
        x, y = self.agent_pos

        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)

        new_pos = (x, y)

        # Check if hit obstacle
        if new_pos in self.obstacles:
            reward = -1
            new_pos = self.agent_pos  # Stay in place
        # Check if reached goal
        elif new_pos == self.goal:
            reward = 10
            done = True
            self.agent_pos = new_pos
            return self._get_state(), reward, done, {}
        else:
            reward = -0.1  # Small penalty for each step

        self.agent_pos = new_pos
        done = False

        return self._get_state(), reward, done, {}

    def _get_state(self) -> int:
        """Convert position to state index."""
        x, y = self.agent_pos
        return x * self.size + y

    def render(self):
        """Print grid."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'

        for obs in self.obstacles:
            grid[obs] = 'X'

        grid[self.goal] = 'G'
        grid[self.agent_pos] = 'A'

        print('\n'.join([' '.join(row) for row in grid]))
        print()


class CartPoleSimple:
    """
    Simplified CartPole Environment

    Discrete version for quick testing.

    States: position bins x velocity bins x angle bins x angular_velocity bins
    Actions: 0 = left, 1 = right
    """

    def __init__(self):
        self.action_space = 2
        self.state_space = 162  # 3 x 3 x 6 x 3 bins

        # Physical parameters
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02

        self.state = None

    def reset(self) -> int:
        """Reset to random state."""
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self._discretize_state()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take step in environment."""
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        # Physics
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.mass_pole * self.length * theta_dot**2 * sintheta) / (self.mass_cart + self.mass_pole)
        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4/3 - self.mass_pole * costheta**2 / (self.mass_cart + self.mass_pole)))
        x_acc = temp - self.mass_pole * self.length * theta_acc * costheta / (self.mass_cart + self.mass_pole)

        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Check termination
        done = bool(
            x < -2.4 or x > 2.4 or
            theta < -0.209 or theta > 0.209
        )

        reward = 1.0 if not done else 0.0

        return self._discretize_state(), reward, done, {}

    def _discretize_state(self) -> int:
        """Convert continuous state to discrete."""
        x, x_dot, theta, theta_dot = self.state

        # Define bins
        x_bins = np.linspace(-2.4, 2.4, 4)[1:-1]
        x_dot_bins = np.linspace(-3, 3, 4)[1:-1]
        theta_bins = np.linspace(-0.209, 0.209, 7)[1:-1]
        theta_dot_bins = np.linspace(-2, 2, 4)[1:-1]

        # Discretize
        x_idx = np.digitize(x, x_bins)
        x_dot_idx = np.digitize(x_dot, x_dot_bins)
        theta_idx = np.digitize(theta, theta_bins)
        theta_dot_idx = np.digitize(theta_dot, theta_dot_bins)

        # Combine into single state
        state = x_idx * 54 + x_dot_idx * 18 + theta_idx * 3 + theta_dot_idx

        return min(state, self.state_space - 1)


if __name__ == "__main__":
    print("=" * 70)
    print("RL UTILITIES EXAMPLES")
    print("=" * 70)

    # Example 1: Replay Buffer
    print("\n1. Replay Buffer")
    print("-" * 70)

    buffer = ReplayBuffer(capacity=1000)

    # Add some transitions
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False

        buffer.push(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample batch
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones = batch
    print(f"Batch shapes: states={states.shape}, actions={actions.shape}")

    # Example 2: GridWorld
    print("\n2. GridWorld Environment")
    print("-" * 70)

    env = GridWorld(size=5, obstacles=[(1, 1), (2, 2)])
    state = env.reset()

    print("Initial state:")
    env.render()

    # Take some actions
    for _ in range(3):
        action = np.random.randint(4)
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}")

        if done:
            print("Goal reached!")
            break

    env.render()

    # Example 3: Reward Normalizer
    print("\n3. Reward Normalization")
    print("-" * 70)

    normalizer = RewardNormalizer()

    rewards = [1.0, 5.0, 2.0, 10.0, 3.0]
    print("Original rewards:", rewards)

    normalized = [normalizer.normalize(r) for r in rewards]
    print("Normalized rewards:", [f"{r:.4f}" for r in normalized])

    print("\n" + "=" * 70)
    print("All utility examples completed!")
    print("=" * 70)
