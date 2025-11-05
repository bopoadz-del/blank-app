"""
Reinforcement Learning Framework

A comprehensive framework for reinforcement learning including:
- Q-Learning (Tabular, DQN, Double DQN, Dueling DQN)
- Policy Gradients (REINFORCE, PPO)
- Actor-Critic (A2C, DDPG)
- Replay buffers
- Environments and reward shaping

Author: ML Framework Team
Version: 1.0.0
"""

# Q-Learning
from .q_learning import (
    TabularQLearning,
    DQN,
    DoubleDQN,
    DuelingDQN,
    QNetwork,
    DuelingQNetwork
)

# Policy Gradients
from .policy_gradient import (
    REINFORCE,
    PPO,
    PolicyNetwork,
    ContinuousPolicyNetwork
)

# Actor-Critic
from .actor_critic import (
    A2C,
    DDPG,
    ActorNetwork,
    CriticNetwork
)

# Utilities
from .utils import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    RewardNormalizer,
    GridWorld,
    CartPoleSimple,
    normalize_reward,
    clip_reward,
    shaped_reward
)

__all__ = [
    # Q-Learning
    'TabularQLearning',
    'DQN',
    'DoubleDQN',
    'DuelingDQN',
    'QNetwork',
    'DuelingQNetwork',

    # Policy Gradients
    'REINFORCE',
    'PPO',
    'PolicyNetwork',
    'ContinuousPolicyNetwork',

    # Actor-Critic
    'A2C',
    'DDPG',
    'ActorNetwork',
    'CriticNetwork',

    # Utilities
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'RewardNormalizer',
    'GridWorld',
    'CartPoleSimple',
    'normalize_reward',
    'clip_reward',
    'shaped_reward',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
