# Reinforcement Learning Framework

A comprehensive Python framework for reinforcement learning, implementing all major RL algorithms from tabular methods to deep RL.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Examples](#examples)
- [API Reference](#api-reference)

## Overview

This framework provides production-ready implementations of:

### Value-Based Methods
- **Tabular Q-Learning**: Classic Q-learning for discrete state-action spaces
- **DQN**: Deep Q-Network with experience replay and target networks
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage functions

### Policy-Based Methods
- **REINFORCE**: Monte Carlo Policy Gradient
- **PPO**: Proximal Policy Optimization (state-of-the-art)

### Actor-Critic Methods
- **A2C**: Advantage Actor-Critic
- **DDPG**: Deep Deterministic Policy Gradient (continuous control)

### Utilities
- Experience Replay Buffers (standard and prioritized)
- Reward shaping and normalization
- Simple environments for testing
- Evaluation metrics

## Installation

### Core Dependencies
```bash
pip install numpy torch
```

### Optional (for visualization and advanced features)
```bash
pip install matplotlib gym
```

## Quick Start

### 1. Tabular Q-Learning (GridWorld)

```python
from reinforcement_learning import TabularQLearning, GridWorld

# Create environment
env = GridWorld(size=5)

# Create Q-learning agent
agent = TabularQLearning(
    n_states=25,
    n_actions=4,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
```

### 2. Deep Q-Network (DQN)

```python
from reinforcement_learning import DQN, ReplayBuffer
import numpy as np

# Create DQN agent
dqn = DQN(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64],
    learning_rate=0.001,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    batch_size=64
)

# Create replay buffer
buffer = ReplayBuffer(capacity=10000)

# Training loop
for episode in range(500):
    state = env.reset()
    episode_reward = 0

    for step in range(200):
        # Select action
        action = dqn.get_action(state)

        # Take step
        next_state, reward, done, _ = env.step(action)

        # Store transition
        buffer.push(state, action, reward, next_state, done)

        # Train if enough samples
        if len(buffer) > 64:
            loss = dqn.train_step(buffer)

        state = next_state
        episode_reward += reward

        if done:
            break

    dqn.decay_epsilon()

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, ε: {dqn.epsilon:.3f}")
```

### 3. Policy Gradient (REINFORCE)

```python
from reinforcement_learning import REINFORCE

# Create REINFORCE agent
agent = REINFORCE(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64],
    learning_rate=0.001,
    discount_factor=0.99
)

# Training loop
for episode in range(1000):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False

    # Collect episode
    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    # Update policy
    loss = agent.update(states, actions, rewards)

    if (episode + 1) % 100 == 0:
        total_reward = sum(rewards)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")
```

### 4. PPO (Proximal Policy Optimization)

```python
from reinforcement_learning import PPO
import numpy as np

# Create PPO agent
ppo = PPO(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64],
    learning_rate=0.0003,
    clip_epsilon=0.2
)

# Collect trajectories
n_steps = 2048
states = np.zeros((n_steps, 4))
actions = np.zeros(n_steps, dtype=int)
rewards = np.zeros(n_steps)
next_states = np.zeros((n_steps, 4))
dones = np.zeros(n_steps)

state = env.reset()
for step in range(n_steps):
    action = ppo.get_action(state)
    next_state, reward, done, _ = env.step(action)

    states[step] = state
    actions[step] = action
    rewards[step] = reward
    next_states[step] = next_state
    dones[step] = done

    state = next_state if not done else env.reset()

# Update policy
metrics = ppo.update(states, actions, rewards, next_states, dones)
print(f"Policy Loss: {metrics['policy_loss']:.4f}")
print(f"Value Loss: {metrics['value_loss']:.4f}")
```

### 5. Actor-Critic (A2C)

```python
from reinforcement_learning import A2C

# Create A2C agent
a2c = A2C(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64],
    learning_rate=0.001,
    value_coef=0.5,
    entropy_coef=0.01
)

# Training loop similar to PPO
for episode in range(1000):
    # Collect batch of transitions
    states, actions, rewards, next_states, dones = collect_batch(env, a2c, n_steps=64)

    # Update
    metrics = a2c.update(states, actions, rewards, next_states, dones)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Actor Loss: {metrics['actor_loss']:.4f}")
```

### 6. DDPG (Continuous Control)

```python
from reinforcement_learning import DDPG, ReplayBuffer

# Create DDPG agent
ddpg = DDPG(
    state_dim=4,
    action_dim=2,
    max_action=2.0,
    learning_rate=0.001
)

buffer = ReplayBuffer(capacity=100000)

# Training loop
for episode in range(500):
    state = env.reset()
    episode_reward = 0

    for step in range(200):
        # Select action with exploration noise
        action = ddpg.get_action(state, noise=0.1)

        # Take step
        next_state, reward, done, _ = env.step(action)

        # Store and train
        buffer.push(state, action, reward, next_state, done)

        if len(buffer) > 64:
            metrics = ddpg.update(buffer, batch_size=64)

        state = next_state
        episode_reward += reward

        if done:
            break

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
```

## Algorithms

### Q-Learning

**Tabular Q-Learning**
- For small discrete state-action spaces
- Simple and effective for grid worlds
- No neural networks required

**Deep Q-Network (DQN)**
- Uses neural network to approximate Q-values
- Experience replay for stable training
- Target network to reduce correlations
- Best for: Discrete action spaces

**Double DQN**
- Reduces overestimation bias
- Uses online network for action selection
- Uses target network for evaluation

**Dueling DQN**
- Separates value and advantage functions
- Better learning in states where action choice doesn't matter
- Architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

### Policy Gradients

**REINFORCE**
- Monte Carlo policy gradient
- Simple and easy to implement
- High variance (use baselines)
- On-policy algorithm

**PPO (Proximal Policy Optimization)**
- State-of-the-art policy gradient method
- Clipped surrogate objective prevents large updates
- Better sample efficiency than REINFORCE
- On-policy algorithm
- Best for: Most RL tasks

### Actor-Critic

**A2C (Advantage Actor-Critic)**
- Synchronous version of A3C
- Uses advantage function to reduce variance
- Both policy and value networks
- On-policy algorithm

**DDPG (Deep Deterministic Policy Gradient)**
- For continuous action spaces
- Off-policy actor-critic
- Deterministic policy
- Uses replay buffer
- Best for: Continuous control (robotics)

## Algorithm Comparison

| Algorithm | Action Space | On/Off-Policy | Sample Efficiency | Stability |
|-----------|--------------|---------------|-------------------|-----------|
| Tabular Q-Learning | Discrete | Off | Medium | High |
| DQN | Discrete | Off | High | Medium |
| Double DQN | Discrete | Off | High | High |
| REINFORCE | Discrete/Continuous | On | Low | Medium |
| PPO | Discrete/Continuous | On | Medium | High |
| A2C | Discrete | On | Medium | Medium |
| DDPG | Continuous | Off | High | Medium |

## Examples

### Complete CartPole Example

```python
from reinforcement_learning import DQN, ReplayBuffer, CartPoleSimple
import numpy as np

env = CartPoleSimple()
agent = DQN(state_dim=4, action_dim=2, epsilon=1.0, epsilon_decay=0.995)
buffer = ReplayBuffer(capacity=10000)

for episode in range(500):
    state = env.reset()
    episode_reward = 0

    for step in range(200):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        buffer.push(state, action, reward, next_state, done)

        if len(buffer) > 64:
            loss = agent.train_step(buffer)

        state = next_state
        episode_reward += reward

        if done:
            break

    agent.decay_epsilon()

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, ε={agent.epsilon:.3f}")

# Save trained agent
agent.save('cartpole_dqn.pth')
```

### Reward Shaping Example

```python
from reinforcement_learning import shaped_reward, RewardNormalizer

# Potential-based reward shaping
def potential(state):
    # Example: distance to goal
    return -np.linalg.norm(state - goal_state)

# Use shaped reward
shaped_r = shaped_reward(state, next_state, reward, gamma=0.99, potential_fn=potential)

# Or normalize rewards
normalizer = RewardNormalizer(gamma=0.99)
normalized_reward = normalizer.normalize(reward)
```

## API Reference

### Q-Learning

#### TabularQLearning
```python
TabularQLearning(
    n_states: int,
    n_actions: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 0.1
)
```
- `get_action(state, eval_mode=False)`: Select action
- `update(state, action, reward, next_state, done)`: Update Q-table
- `decay_epsilon()`: Decay exploration rate

#### DQN
```python
DQN(
    state_dim: int,
    action_dim: int,
    hidden_dims: list = [64, 64],
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    epsilon: float = 1.0
)
```
- `get_action(state, eval_mode=False)`: Select action
- `train_step(replay_buffer, batch_size=64)`: Training step
- `save(filepath)`, `load(filepath)`: Save/load model

### Policy Gradients

#### REINFORCE
```python
REINFORCE(
    state_dim: int,
    action_dim: int,
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    continuous: bool = False
)
```
- `get_action(state, deterministic=False)`: Select action
- `update(states, actions, rewards)`: Update policy

#### PPO
```python
PPO(
    state_dim: int,
    action_dim: int,
    learning_rate: float = 0.0003,
    clip_epsilon: float = 0.2,
    epochs: int = 10
)
```
- `get_action(state, deterministic=False)`: Select action
- `update(states, actions, rewards, next_states, dones)`: PPO update
- Returns metrics: policy_loss, value_loss, entropy

### Actor-Critic

#### A2C
```python
A2C(
    state_dim: int,
    action_dim: int,
    learning_rate: float = 0.001,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
)
```
- `get_action(state)`: Select action
- `update(states, actions, rewards, next_states, dones)`: Update

#### DDPG
```python
DDPG(
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
    learning_rate: float = 0.001,
    tau: float = 0.005
)
```
- `get_action(state, noise=0.1)`: Select action with exploration
- `update(replay_buffer, batch_size=64)`: Training step

### Utilities

#### ReplayBuffer
```python
ReplayBuffer(capacity: int)
```
- `push(state, action, reward, next_state, done)`: Add transition
- `sample(batch_size)`: Sample batch
- `__len__()`: Get buffer size

#### GridWorld
```python
GridWorld(size: int = 5, goal: tuple = None, obstacles: list = None)
```
- `reset()`: Reset environment
- `step(action)`: Take action (0=up, 1=down, 2=left, 3=right)
- `render()`: Display grid

## Best Practices

### 1. Start Simple
```python
# Always start with tabular methods or simple DQN
agent = TabularQLearning(...)  # For small problems
agent = DQN(...)  # For larger state spaces
```

### 2. Tune Hyperparameters
```python
# Important hyperparameters
learning_rate = 0.0003  # Lower for stability
discount_factor = 0.99  # Higher for long-term planning
epsilon = 1.0  # Start with full exploration
epsilon_decay = 0.995  # Decay slowly
```

### 3. Use Replay Buffers
```python
# Always use replay buffer for off-policy methods
buffer = ReplayBuffer(capacity=100000)  # Large capacity
# Only train when buffer has enough samples
if len(buffer) > batch_size:
    agent.train_step(buffer)
```

### 4. Monitor Training
```python
# Track key metrics
episode_rewards = []
losses = []

if (episode + 1) % 100 == 0:
    avg_reward = np.mean(episode_rewards[-100:])
    print(f"Avg Reward (last 100): {avg_reward:.2f}")
```

### 5. Save and Load Models
```python
# Save best model
if episode_reward > best_reward:
    best_reward = episode_reward
    agent.save('best_model.pth')

# Load for evaluation
agent.load('best_model.pth')
```

## File Structure

```
reinforcement-learning/
├── __init__.py              # Package initialization
├── q_learning.py            # Q-learning algorithms
├── policy_gradient.py       # REINFORCE, PPO
├── actor_critic.py          # A2C, DDPG
├── utils.py                 # Buffers, environments, rewards
└── README.md                # This file
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

## License

This framework is provided as-is for educational and commercial use.

---

**Happy Learning! 🤖**
