---
title: Reinforcement Learning
tags: [reinforcement-learning, theory, fundamentals]
created: 2026-01-16
related: [Q-Learning, Deep Q-Network, Self-Play, Policy vs Value Functions]
---

# Reinforcement Learning

## Overview

**Reinforcement Learning (RL)** is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards (or penalties) for its actions and learns to maximize cumulative reward over time.

## Key Components

### Agent
The learner or decision-maker that interacts with the environment.

### Environment
The world with which the agent interacts. In this project: [[CuttleEnvironment]]

### State (s)
A representation of the current situation. In this project: [[Observation Space]] (468 boolean features)

### Action (a)
A choice the agent can make. In this project: Discrete actions via [[Action System]]

### Reward (r)
Feedback signal indicating how good/bad an action was. Guides learning toward desirable behaviors.

### Policy (π)
A mapping from states to actions. The agent's strategy for choosing actions.
- **Deterministic**: π(s) = a (always same action for same state)
- **Stochastic**: π(a|s) = probability (probabilistic action selection)

## The RL Loop

```
Agent observes state s_t
  ↓
Agent selects action a_t (according to policy π)
  ↓
Environment transitions to state s_{t+1}
  ↓
Agent receives reward r_t
  ↓
Agent updates policy based on (s_t, a_t, r_t, s_{t+1})
  ↓
Repeat
```

## Mathematical Framework

### Markov Decision Process (MDP)

RL problems are often formalized as MDPs:

- **States** S: Set of possible states
- **Actions** A: Set of possible actions
- **Transition Function** P(s'|s,a): Probability of next state given current state and action
- **Reward Function** R(s,a,s'): Expected reward for transition
- **Discount Factor** γ: How much we value future rewards (0 ≤ γ ≤ 1)

### Return (G_t)

Total future reward from time t:

```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
```

The discount factor γ makes immediate rewards more valuable than distant ones.

## Value Functions

### State Value Function V^π(s)

Expected return starting from state s following policy π:
```
V^π(s) = E_π[G_t | s_t = s]
```

### Action Value Function Q^π(s,a)

Expected return starting from state s, taking action a, then following policy π:
```
Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
```

Also called **Q-function** or **Q-values**. See [[Q-Learning]] for how Q-values are learned.

## Main Approaches

### 1. Value-Based Methods

Learn value functions (V or Q), derive policy from values.
- **[[Q-Learning]]**: Learn Q(s,a) directly
- **[[Deep Q-Network]]**: Use neural networks to approximate Q(s,a)
- Example: This project uses DQN

### 2. Policy-Based Methods

Learn policy directly without value functions.
- **Policy Gradient**: Optimize policy parameters via gradient ascent
- **REINFORCE**: Simple policy gradient algorithm

### 3. Actor-Critic Methods

Combine value functions and policy learning.
- **Actor**: Policy (what to do)
- **Critic**: Value function (how good is it)

See [[Policy vs Value Functions]] for comparison.

## On-Policy vs Off-Policy

### On-Policy

Learn about the policy being used to collect data.
- Examples: SARSA, REINFORCE
- More stable but less sample efficient

### Off-Policy

Learn about a different policy than the one used to collect data.
- Examples: [[Q-Learning]], [[Deep Q-Network]]
- More sample efficient but can be unstable

See [[On-Policy vs Off-Policy]] for details.

## Exploration vs Exploitation

**Exploration**: Try new actions to discover better strategies
**Exploitation**: Use current best-known actions

Balancing is crucial:
- Too much exploration: Slow learning, random behavior
- Too much exploitation: Stuck in local optima, miss better strategies

### Exploration Strategies

1. **[[Epsilon-Greedy Exploration]]**: Random action with probability ε
2. **Upper Confidence Bound (UCB)**: Balance uncertainty and value
3. **Thompson Sampling**: Probabilistic action selection

## Challenges in RL

### 1. Sample Efficiency
RL often requires many interactions with environment.

### 2. Credit Assignment
Determining which actions led to rewards (especially delayed rewards).

### 3. Exploration
Finding good actions while avoiding bad ones.

### 4. Stability
RL can be unstable, especially with function approximation.

### 5. Reward Engineering
Designing effective reward signals (see [[Reward Engineering]]).

## RL in This Project

- **Algorithm**: [[Deep Q-Network]] (value-based, off-policy)
- **Training**: [[Self-Play]] - agents train by playing against themselves
- **Experience Storage**: [[Experience Replay]] buffer
- **Exploration**: [[Epsilon-Greedy Exploration|Epsilon-greedy]] (ε decays during training)
- **Environment**: [[CuttleEnvironment]] (Gymnasium-compatible)

## Related Concepts

- [[Q-Learning]] - Value-based RL algorithm
- [[Deep Q-Network]] - Neural networks for Q-learning
- [[Self-Play]] - Training methodology
- [[Experience Replay]] - Learning from past experiences
- [[Reward Engineering]] - Designing reward signals
- [[Policy vs Value Functions]] - Different RL approaches
- [[On-Policy vs Off-Policy]] - Learning algorithm categories

## Further Reading

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Arulkumaran, K., et al. (2017). "Deep Reinforcement Learning: A Brief Survey." *IEEE Signal Processing Magazine*.

---
*RL fundamentals for thesis reference*
