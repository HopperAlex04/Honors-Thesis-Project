---
title: Q-Learning
tags: [reinforcement-learning, q-learning, value-based]
created: 2026-01-16
related: [Deep Q-Network, Reinforcement Learning, Experience Replay]
---

# Q-Learning

## Overview

**Q-Learning** is an off-policy, value-based reinforcement learning algorithm that learns the optimal action-value function Q*(s,a) directly. It's model-free (doesn't require knowing environment dynamics) and can learn optimal policies even while following an exploratory policy.

## Q-Function

The Q-function Q(s,a) represents the expected cumulative reward of:
1. Starting in state s
2. Taking action a
3. Then following the optimal policy thereafter

```
Q*(s,a) = max_π E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, a_t = a, π]
```

Where:
- γ (gamma) is the discount factor
- R_t is the reward at time t

## Bellman Equation

The optimal Q-function satisfies the Bellman optimality equation:

```
Q*(s,a) = E[r + γ max_{a'} Q*(s', a') | s, a]
```

This states: the optimal Q-value equals the immediate reward plus the discounted value of the best action in the next state.

## Q-Learning Algorithm

### Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s', a') - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate (step size)
- **r**: Immediate reward
- **γ (gamma)**: Discount factor
- **s'**: Next state
- **max_{a'} Q(s', a')**: Best Q-value in next state

The term in brackets `[r + γ max_{a'} Q(s', a') - Q(s,a)]` is the **Temporal Difference (TD) error**.

### Pseudocode

```
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize state s
    For each step:
        Choose action a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe reward r and next state s'
        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s', a') - Q(s,a)]
        s ← s'
```

## Key Properties

### Off-Policy Learning

Q-Learning is **off-policy**: it learns the optimal Q-function (Q*) while following any policy (including an exploratory one). This allows:
- Learning from any experiences (replay buffer)
- Learning optimal policy while exploring

Compare to [[On-Policy vs Off-Policy|on-policy]] methods like SARSA that learn about the policy being followed.

### Model-Free

Q-Learning doesn't require knowledge of:
- Transition probabilities P(s'|s,a)
- Reward function R(s,a)

It learns directly from experience (s, a, r, s') tuples.

## Limitations of Tabular Q-Learning

Traditional Q-Learning uses a table to store Q(s,a) for every state-action pair:
- **Problem**: Doesn't scale to large/continuous state spaces
- **Example**: 468-dimension state space (this project) = 2^468 possible states (intractable!)

## Function Approximation

To handle large state spaces, we approximate Q(s,a) using:
- [[Neural Network Basics|Neural networks]] → [[Deep Q-Network]]
- Linear function approximators
- Other function approximators

This project uses [[Deep Q-Network|DQN]]: neural networks to approximate Q-values.

## Hyperparameters

### Learning Rate (α)

Step size for Q-value updates:
- **Too high**: Unstable learning, overshoot optimal values
- **Too low**: Slow convergence
- In this project: Learning rate managed by [[Optimization Algorithms|optimizer]] (e.g., Adam)

### Discount Factor (γ)

How much we value future rewards:
- **γ = 0**: Only care about immediate reward (myopic)
- **γ = 1**: Equal weight to all future rewards (undiscounted)
- **Typical**: 0.9 - 0.99
- In this project: γ = 0.90 (see [[Hyperparameters]])

### Exploration

Need to explore while learning Q-values:
- **[[Epsilon-Greedy Exploration]]**: Random action with probability ε
- **ε decay**: Start high (explore), decay to low (exploit)
- In this project: ε_start = 0.90, ε_end = 0.05

## Experience Replay

Instead of learning from single experiences sequentially, store experiences in a replay buffer and sample batches:
- **Benefits**: 
  - Breaks correlation between consecutive experiences
  - More sample efficient (reuse experiences)
  - More stable learning
- See [[Experience Replay]] for details

## Target Networks

In [[Deep Q-Network]], use a separate target network for Q(s', a'):
- Reduces moving target problem
- More stable learning
- Updated periodically (not every step)

## Q-Learning in This Project

This project uses **Deep Q-Network (DQN)**, which combines:
- [[Q-Learning]] update rule
- [[Neural Network Basics|Neural networks]] for function approximation
- [[Experience Replay]] buffer
- Target networks
- [[Epsilon-Greedy Exploration|Epsilon-greedy]] exploration

See [[Deep Q-Network]] for implementation details.

## Related Concepts

- [[Deep Q-Network]] - Q-Learning with neural networks
- [[Reinforcement Learning]] - Broader RL context
- [[Experience Replay]] - Learning from stored experiences
- [[Epsilon-Greedy Exploration]] - Exploration strategy
- [[Bellman Equation]] - Foundation for Q-Learning

## Further Reading

- Watkins, C. J. C. H., & Dayan, P. (1992). "Q-learning." *Machine Learning*.
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.

---
*Q-Learning algorithm for thesis reference*
