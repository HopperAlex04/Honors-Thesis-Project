---
title: Epsilon-Greedy Exploration
tags: [reinforcement-learning, exploration, training]
created: 2026-01-16
related: [Reinforcement Learning, Deep Q-Network, Q-Learning]
---

# Epsilon-Greedy Exploration

## Overview

**Epsilon-Greedy** (ε-greedy) is a simple but effective exploration strategy in reinforcement learning. With probability ε (epsilon), the agent takes a random action (exploration); otherwise, it takes the best-known action (exploitation).

## The Exploration-Exploitation Dilemma

### Problem

RL agents face a fundamental trade-off:

- **Exploitation**: Use current best-known actions (greedy)
  - Benefit: Maximizes immediate reward
  - Risk: Miss better strategies
  
- **Exploration**: Try new actions to discover better strategies
  - Benefit: May discover optimal actions
  - Risk: Takes suboptimal actions

### Solution: Balance

Need to balance exploration and exploitation throughout learning.

## Epsilon-Greedy Strategy

### Algorithm

```
With probability ε:
    Take random action (exploration)
Otherwise (with probability 1-ε):
    Take action with highest Q-value (exploitation)
```

### Pseudocode

```python
import random

def select_action(state, epsilon, q_values):
    if random.random() < epsilon:
        # Exploration: random action
        action = random.randint(0, num_actions - 1)
    else:
        # Exploitation: best action
        action = q_values.argmax()
    return action
```

## Epsilon (ε) Values

### High ε (e.g., 0.9)

- **More exploration**: Many random actions
- **Use**: Early in training (agent knows little)
- **Benefit**: Discovers diverse strategies
- **Cost**: Slow learning (many random actions)

### Low ε (e.g., 0.05)

- **More exploitation**: Mostly best actions
- **Use**: Later in training (agent learned good strategies)
- **Benefit**: Faster learning (uses learned knowledge)
- **Cost**: May miss better strategies

### Typical Schedule

Start with high ε, decay to low ε during training:

```
ε(t) = ε_start * (ε_end / ε_start)^(t / decay_steps)
```

Or linear decay:

```
ε(t) = max(ε_end, ε_start - (ε_start - ε_end) * t / decay_steps)
```

## Epsilon Decay

Gradually reduce ε during training to shift from exploration to exploitation.

### Exponential Decay

```
ε(t) = ε_start * decay_rate^t
```

### Linear Decay

```
ε(t) = ε_start - (ε_start - ε_end) * t / total_steps
```

### Step Decay

```
ε(t) = ε_start if t < T1
       ε_mid if T1 ≤ t < T2
       ε_end if t ≥ T2
```

## Hyperparameters

### Initial Epsilon (ε_start)

Starting exploration rate:
- **Typical**: 0.9 - 1.0
- **This project**: 0.90 (see [[Hyperparameters]])

### Final Epsilon (ε_end)

Final exploration rate:
- **Typical**: 0.0 - 0.1
- **This project**: 0.05 (see [[Hyperparameters]])
- **Reason**: Keep small exploration (5%) for robustness

### Decay Rate / Decay Steps

How quickly to decay ε:
- **Decay steps**: Number of steps/episodes to decay
- **This project**: 28,510 episodes (see [[Hyperparameters]])

## Example Schedule

For ε_start = 0.9, ε_end = 0.05, decay_steps = 28,510:

| Episode | Epsilon |
|---------|---------|
| 0       | 0.90    |
| 5,000   | ~0.70   |
| 10,000  | ~0.52   |
| 20,000  | ~0.29   |
| 28,510  | 0.05    |

## Benefits

### 1. Simplicity

Easy to implement and understand:
- Single parameter (ε)
- Clear interpretation
- Works well in practice

### 2. Explicit Control

Control exploration-exploitation trade-off:
- Adjust ε to prioritize exploration or exploitation
- Adapt during training (decay schedule)

### 3. Works Well

Effective for many RL problems:
- Balances exploration and exploitation
- Commonly used in practice

## Limitations

### 1. Uniform Random Exploration

Random actions sampled uniformly:
- **Problem**: May explore obviously bad actions
- **Alternative**: Upper Confidence Bound (UCB) explores uncertain actions

### 2. Fixed Schedule

Epsilon schedule predetermined:
- **Problem**: May decay too fast or too slow
- **Alternative**: Adaptive exploration (exploration bonus, curiosity)

### 3. Ignores Uncertainty

Doesn't account for uncertainty in Q-values:
- **Problem**: May not explore where most uncertain
- **Alternative**: Thompson Sampling, Bayesian RL

## Alternatives

### 1. Upper Confidence Bound (UCB)

Explores actions with high uncertainty/upper bound:
- `action = argmax(Q(s,a) + c * sqrt(log(N(s))/N(s,a)))`
- Balances value and uncertainty

### 2. Thompson Sampling

Probabilistic action selection:
- Samples from posterior distribution over Q-values
- Natural exploration of uncertainty

### 3. Exploration Bonus

Reward for visiting novel states:
- `reward = environment_reward + β * novelty(s)`
- Encourages exploration of new states

## Epsilon-Greedy in This Project

### Configuration

From `hyperparams_config.json`:
- **ε_start**: 0.90 (high initial exploration)
- **ε_end**: 0.05 (small final exploration)
- **ε_decay**: 28,510 episodes (gradual decay)

### Implementation

Used in [[Training Process]]:
- Agent selects actions during training
- Epsilon decays over episodes
- Balances exploration (early) and exploitation (later)

### Integration with DQN

- **Exploration**: Random actions (probability ε)
- **Exploitation**: Best Q-value actions (probability 1-ε)
- **Learning**: Agent updates Q-values from both exploration and exploitation experiences

## Related Concepts

- [[Reinforcement Learning]] - Exploration in RL context
- [[Deep Q-Network]] - Uses epsilon-greedy for action selection
- [[Q-Learning]] - Can use epsilon-greedy exploration
- [[Training Process]] - Implementation in this project

## Further Reading

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. (Chapter 2: Multi-armed Bandits)

---
*Epsilon-greedy exploration strategy used in this project*
