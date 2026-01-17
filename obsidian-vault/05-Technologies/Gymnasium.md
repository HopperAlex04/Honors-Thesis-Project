---
title: Gymnasium
tags: [technology, framework, reinforcement-learning, gymnasium]
created: 2026-01-16
related: [CuttleEnvironment, Reinforcement Learning]
---

# Gymnasium

## Overview

**Gymnasium** (formerly OpenAI Gym) is a Python library that provides a standard interface for reinforcement learning environments. It defines a common API for environments, making it easy to develop, test, and compare RL algorithms.

## History

- **OpenAI Gym**: Original version (2016-2022)
- **Gymnasium**: Fork maintained by Farama Foundation (2022-present)
  - Same API and functionality
  - Better maintenance, more environments
  - Community-driven development

## Core Interface

### Environment API

All Gymnasium environments follow a standard interface:

#### 1. Initialization

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset()
```

#### 2. Step Function

The core interaction method:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

**Returns**:
- **observation**: New state after taking action
- **reward**: Reward for this step
- **terminated**: Whether episode ended naturally (e.g., game won)
- **truncated**: Whether episode ended due to limit (e.g., max steps)
- **info**: Additional debugging information

#### 3. Reset

Restart environment:

```python
observation, info = env.reset()
```

Returns initial observation.

### Observation and Action Spaces

#### Observation Space

Describes the structure of observations:

```python
print(env.observation_space)
# Example: Box(0, 1, (468,), dtype=float32)
```

Common space types:
- **Box**: Continuous values in a box (n-dimensional array)
- **Discrete**: Integer from 0 to n-1
- **MultiDiscrete**: Multiple discrete spaces
- **Dict**: Dictionary of spaces (used in this project)

#### Action Space

Describes possible actions:

```python
print(env.action_space)
# Example: Discrete(100)  # 100 possible actions
```

Common space types:
- **Discrete**: Integer action (0, 1, 2, ...)
- **Box**: Continuous action vector
- **MultiDiscrete**: Multiple discrete actions

## Custom Environments

### Creating a Custom Environment

To create a custom environment (like [[CuttleEnvironment]]), subclass `gym.Env`:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "zones": spaces.Box(0, 1, (364,), dtype=bool),
            "stack": spaces.Box(0, 1, (52,), dtype=bool),
            "effect_shown": spaces.Box(0, 1, (52,), dtype=bool),
        })
        self.action_space = spaces.Discrete(num_actions)
    
    def reset(self, seed=None, options=None):
        # Initialize environment, return initial observation
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        # Execute action, update state
        # Compute reward
        # Check if episode done
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminal()
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
```

### Required Methods

1. **`__init__`**: Initialize environment, define spaces
2. **`reset`**: Reset environment, return initial observation
3. **`step`**: Take action, return (observation, reward, terminated, truncated, info)
4. **`render`** (optional): Visualize environment

## Gymnasium in This Project

### CuttleEnvironment

The [[CuttleEnvironment]] implements the Gymnasium interface:

```python
class CuttleEnvironment:
    def __init__(self):
        # Define observation_space (Dict of boolean arrays)
        # Define action_space (Discrete actions)
    
    def reset(self, seed=None, options=None):
        # Initialize game state
        # Return initial observation
    
    def step(self, action):
        # Execute game action
        # Update game state
        # Compute reward
        # Check if game ended
        return observation, reward, terminated, truncated, info
```

### Observation Space

See [[Observation Space]] for details on the observation structure:

```python
self.observation_space = spaces.Dict({
    "Current Zones": {
        "Hand": spaces.Box(0, 1, (52,), dtype=bool),
        "Field": spaces.Box(0, 1, (52,), dtype=bool),
        "Revealed": spaces.Box(0, 1, (52,), dtype=bool),
    },
    "Off Zones": {
        "Hand": spaces.Box(0, 1, (52,), dtype=bool),
        "Field": spaces.Box(0, 1, (52,), dtype=bool),
        "Revealed": spaces.Box(0, 1, (52,), dtype=bool),
    },
    "Deck": spaces.Box(0, 1, (52,), dtype=bool),
    "Scrap": spaces.Box(0, 1, (52,), dtype=bool),
    "Stack": spaces.Box(0, 1, (52,), dtype=bool),
    "Effect-Shown": spaces.Box(0, 1, (52,), dtype=bool),
})
```

### Action Space

Discrete actions via [[Action System|ActionRegistry]]:

```python
self.action_space = spaces.Discrete(num_actions)
```

Where `num_actions` is the total number of valid game actions.

## Benefits of Gymnasium API

### 1. Standardization

All environments follow same interface:
- Easy to switch between environments
- Algorithms work with any Gymnasium environment
- Consistent testing and evaluation

### 2. Compatibility

Works with RL libraries:
- **Stable-Baselines3**: RL algorithms
- **Ray RLlib**: Distributed RL
- Custom implementations (like this project)

### 3. Reproducibility

- **Seed parameter**: `env.reset(seed=42)` for reproducibility
- **Options**: Additional configuration

### 4. Observation/Action Spaces

Clear definition of:
- What agent can observe
- What actions agent can take
- Enables automatic preprocessing

## Common Operations

### Environment Loop

```python
observation, info = env.reset()
done = False
while not done:
    action = agent.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    agent.update(observation, reward, done)
```

### Wrappers

Modify environments without changing code:

```python
from gymnasium.wrappers import TimeLimit, ClipAction

env = CustomEnv()
env = TimeLimit(env, max_episode_steps=200)
env = ClipAction(env)
```

Common wrappers:
- **TimeLimit**: Enforce maximum episode length
- **ClipAction**: Clip actions to valid range
- **NormalizeObservation**: Normalize observations

## Related Concepts

- [[CuttleEnvironment]] - Custom environment implementation
- [[Reinforcement Learning]] - RL framework using Gymnasium
- [[Observation Space]] - Observation structure
- [[Action System]] - Action space definition

## Resources

- **Official Docs**: https://gymnasium.farama.org/
- **GitHub**: https://github.com/Farama-Foundation/Gymnasium
- **Environments**: https://gymnasium.farama.org/environments/overview/

---
*Gymnasium RL environment interface used in this project*
