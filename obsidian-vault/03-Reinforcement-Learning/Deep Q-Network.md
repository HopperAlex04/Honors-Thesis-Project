---
title: Deep Q-Network
tags: [reinforcement-learning, dqn, deep-learning, neural-networks]
created: 2026-01-16
related: [Q-Learning, Neural Network Architecture, Experience Replay, Self-Play]
---

# Deep Q-Network (DQN)

## Overview

**Deep Q-Network (DQN)** combines [[Q-Learning]] with [[Neural Network Basics|deep neural networks]] to approximate Q-values for large state spaces. Developed by Mnih et al. (2015), DQN enabled RL to work with high-dimensional inputs (like raw pixels or complex game states).

## Motivation

Traditional [[Q-Learning]] uses a table Q(s,a) - doesn't scale to large/continuous state spaces.

**Solution**: Approximate Q(s,a) using a neural network:
```
Q(s,a; θ) ≈ Q*(s,a)
```

Where θ are network parameters (weights and biases).

## Architecture

### Input

State representation (in this project: [[Observation Space]] - 468 boolean features)

### Output

Q-values for all possible actions:
- If 100 possible actions → network outputs 100 Q-values
- Action with highest Q-value is the optimal action

### Network Structure

In this project (see [[Neural Network Architecture]]):
```
Input (468 dims) 
  → Linear(468 → 512) + ReLU
  → Linear(512 → 256) + ReLU
  → Linear(256 → num_actions)  [no activation - unbounded Q-values]
```

## Training Process

### 1. Collect Experience

Agent interacts with environment using current policy (e.g., [[Epsilon-Greedy Exploration|epsilon-greedy]]):
```
(s_t, a_t, r_t, s_{t+1})
```

### 2. Store in Replay Buffer

Add experience to [[Experience Replay]] buffer:
```
D = {(s, a, r, s', done), ...}
```

### 3. Sample Batch

Randomly sample batch from buffer (breaks correlation):
```
Batch = {(s_i, a_i, r_i, s'_i, done_i)} for i in batch
```

### 4. Compute Targets

For each experience in batch:
```
y_i = r_i + γ max_{a'} Q(s'_i, a'; θ_target)   if not done
y_i = r_i                                        if done
```

Where θ_target are parameters of the **target network** (see below).

### 5. Compute Loss

Mean Squared Error between predictions and targets:
```
L(θ) = (1/|batch|) Σ (y_i - Q(s_i, a_i; θ))²
```

### 6. Update Network

Backpropagate loss, update parameters θ using gradient descent:
```
θ ← θ - α ∇_θ L(θ)
```

Where α is learning rate (managed by [[Optimization Algorithms|optimizer]]).

### 7. Update Target Network

Periodically copy main network to target network:
```
θ_target ← θ  (every N steps)
```

## Key Innovations

### 1. Experience Replay

Store past experiences in buffer, sample randomly:
- **Breaks correlation**: Consecutive experiences are correlated; random sampling decorrelates
- **Sample efficiency**: Reuse experiences multiple times
- See [[Experience Replay]] for details

### 2. Target Network

Separate network with frozen parameters for computing targets:
- **Problem**: Q-values are moving target (we're learning them while using them)
- **Solution**: Use target network for Q(s', a') in update, update it periodically
- **Effect**: More stable learning

### 3. Frame Stacking / State Representation

For complex observations, design effective state representation:
- In this project: [[Observation Space]] - boolean arrays for all zones
- Other projects: Stack consecutive frames for temporal information

## Hyperparameters

Key hyperparameters (see [[Hyperparameters]] for project config):

- **Learning Rate**: 3e-5 (step size for weight updates)
- **Gamma (γ)**: 0.90 (discount factor for future rewards)
- **Epsilon**: Starts at 0.90, decays to 0.05 (exploration rate)
- **Replay Buffer Size**: 30,000 experiences
- **Batch Size**: 128 experiences per update
- **Target Update Frequency**: Every 500 steps

## Challenges & Solutions

### 1. Instability

**Problem**: Deep RL training can be unstable.

**Solutions**:
- Target networks (see above)
- Experience replay (decorrelation)
- [[Gradient Clipping]] (prevent exploding gradients)
- [[Learning Rate Scheduling]] (adaptive learning rates)

### 2. Overestimation Bias

**Problem**: max operator causes overestimation of Q-values.

**Solutions** (not used in basic DQN):
- Double DQN: Use separate networks for action selection and evaluation
- Dueling DQN: Separate value and advantage estimation

### 3. Exploration

**Problem**: Need to explore while learning.

**Solution**: [[Epsilon-Greedy Exploration|Epsilon-greedy]] - random actions with probability ε

### 4. Reward Scaling

**Problem**: Rewards can be large or small, affecting learning.

**Solution**: [[Reward Engineering]] - scale rewards appropriately (in this project: 1.0 for win, -1.0 for loss, -0.5 for draw)

## DQN Variants

- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage
- **Prioritized Experience Replay**: Sample important experiences more often
- **Rainbow DQN**: Combines multiple improvements

This project uses **vanilla DQN** (basic implementation).

## DQN in This Project

- **Environment**: [[CuttleEnvironment]] (card game)
- **Observation**: [[Observation Space]] (468 boolean features)
- **Network**: [[Neural Network Architecture]] (512→256→num_actions)
- **Training**: [[Self-Play]] (agents play against themselves)
- **Framework**: [[PyTorch]] for neural network implementation

## Related Concepts

- [[Q-Learning]] - Foundation algorithm
- [[Neural Network Architecture]] - Network structure in this project
- [[Experience Replay]] - Learning from stored experiences
- [[Self-Play]] - Training methodology
- [[Epsilon-Greedy Exploration]] - Exploration strategy
- [[Reward Engineering]] - Reward design

## Further Reading

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
- Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." *arXiv*.

---
*DQN algorithm - core method used in this project*
