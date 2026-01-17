---
title: Reward Engineering
tags: [reinforcement-learning, rewards, training]
created: 2026-01-16
related: [Reinforcement Learning, Deep Q-Network, Training Process]
---

# Reward Engineering

## Overview

**Reward Engineering** is the design and tuning of reward signals to guide reinforcement learning agents toward desired behaviors. The reward function is crucial - it shapes what the agent learns and how it behaves.

## The Reward Signal

### Definition

Rewards are scalar signals `r_t` that indicate:
- **Positive reward**: Good action/state (encourage)
- **Negative reward**: Bad action/state (discourage)
- **Zero reward**: Neutral (neither good nor bad)

### Purpose

Rewards serve as the **only learning signal** in RL:
- Agent learns to maximize cumulative reward
- Reward shape determines agent behavior
- Poor reward design → poor agent behavior

## Reward Function Design

### Terminal Rewards

Rewards at episode end (game outcome):

#### Win/Loss/Draw

- **Win**: `r = +1.0` (positive, encourages winning)
- **Loss**: `r = -1.0` (negative, discourages losing)
- **Draw**: `r = -0.5` (negative but less than loss, discourages draws)

In this project:
- Win: `REWARD_WIN = 1.0`
- Loss: `REWARD_LOSS = -1.0`
- Draw: `REWARD_DRAW = -0.5` (penalize passive play)

### Intermediate Rewards

Rewards during episode (not just at end):

#### Score-Based Rewards

Reward for increasing score:
```
r = SCORE_REWARD_SCALE * score_change
```

- Encourages scoring (intermediate progress)
- Helps credit assignment (which actions led to score)
- **Scale**: Small (e.g., 0.01) to prevent Q-value explosion

In this project: `SCORE_REWARD_SCALE = 0.01`

#### Gap-Based Rewards

Reward for increasing score gap (your score - opponent score):
```
r = GAP_REWARD_SCALE * gap_change
```

- Encourages outperforming opponent
- Can help in competitive settings
- **Scale**: Often smaller than score reward

In this project: `GAP_REWARD_SCALE = 0.005` (half of score reward scale)

#### Penalty for Undesired Behavior

Negative rewards for bad actions:
- Illegal moves: `r = -1.0` (strong penalty)
- Invalid actions: Discourage via environment rules

## Reward Scaling

### Problem

Rewards can have different magnitudes:
- Too large: Q-values explode, unstable learning
- Too small: Slow learning, small gradients

### Solution

Scale rewards to appropriate range:

```python
# Scale terminal rewards to [-1, 1]
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_DRAW = -0.5

# Scale intermediate rewards (much smaller)
SCORE_REWARD_SCALE = 0.01  # Score changes are 0.01 per point
GAP_REWARD_SCALE = 0.005   # Gap changes are 0.005 per point
```

### Guidelines

- **Terminal rewards**: Larger magnitude (±1.0)
- **Intermediate rewards**: Small magnitude (0.01, 0.005)
- **Total reward range**: Typically [-10, 10] or smaller for stability

## Reward Shaping

### Definition

Modifying rewards to guide learning:
- **Sparse rewards**: Only terminal rewards (win/loss)
- **Dense rewards**: Rewards throughout episode (score changes)

### Sparse Rewards

Only reward at episode end:
- **Advantage**: Simple, clear objective
- **Disadvantage**: Hard credit assignment (which actions led to win?)

### Dense Rewards

Reward throughout episode:
- **Advantage**: Helps credit assignment, faster learning
- **Disadvantage**: Must design carefully (can lead to wrong behavior)

### Example: This Project

**Sparse component**: Terminal rewards (win/loss/draw)
**Dense component**: Score-based rewards (intermediate progress)

Combination: Dense rewards for learning, sparse rewards for final objective.

## Reward Engineering Principles

### 1. Align with Objective

Rewards should match desired behavior:
- Goal: Win games → reward winning
- Goal: High score → reward scoring
- **Warning**: Agent optimizes reward, not necessarily your true goal!

### 2. Sparse vs Dense Trade-off

- **Sparse**: Closer to true objective, but harder to learn
- **Dense**: Easier to learn, but must align with true objective

### 3. Scale Appropriately

- Terminal rewards: Larger magnitude
- Intermediate rewards: Smaller magnitude
- Total range: Reasonable for Q-value stability

### 4. Avoid Reward Hacking

Agent may exploit reward function:
- **Example**: Agent finds loophole to maximize reward without achieving goal
- **Solution**: Design robust rewards, test agent behavior

### 5. Balance Rewards

Different reward components should have appropriate weights:
- Score reward vs gap reward
- Terminal reward vs intermediate reward

## Reward Engineering in This Project

### Reward Structure

From `training.py` constants:

```python
REWARD_WIN = 1.0              # Terminal: Win
REWARD_LOSS = -1.0            # Terminal: Loss
REWARD_DRAW = -0.5            # Terminal: Draw (penalize passive play)
REWARD_INTERMEDIATE = 0.0     # Base for intermediate steps
SCORE_REWARD_SCALE = 0.01     # Scale for score changes
GAP_REWARD_SCALE = 0.005      # Scale for gap changes (half of score scale)
```

### Design Rationale

1. **Terminal rewards**: Clear signal for game outcome
   - Win: Positive (encourage)
   - Loss: Negative (discourage)
   - Draw: Negative but less (discourage passive play)

2. **Score-based rewards**: Guide intermediate learning
   - Small scale (0.01) to prevent Q-value explosion
   - Helps credit assignment

3. **Gap-based rewards**: Encourage outperforming opponent
   - Smaller scale (0.005) than score rewards
   - Secondary to scoring

### Reward Computation

```python
# Terminal rewards
if game_won:
    reward = REWARD_WIN
elif game_lost:
    reward = REWARD_LOSS
elif game_draw:
    reward = REWARD_DRAW
else:
    # Intermediate rewards
    reward = REWARD_INTERMEDIATE
    reward += SCORE_REWARD_SCALE * score_change
    reward += GAP_REWARD_SCALE * gap_change
```

## Common Pitfalls

### 1. Reward Hacking

Agent finds loophole:
- **Example**: Agent exploits reward function without achieving true goal
- **Solution**: Test agent behavior, refine rewards

### 2. Unbalanced Rewards

One reward component dominates:
- **Problem**: Agent ignores other objectives
- **Solution**: Balance reward scales

### 3. Sparse Only

Only terminal rewards:
- **Problem**: Hard credit assignment, slow learning
- **Solution**: Add intermediate rewards (carefully)

### 4. Too Dense

Too many intermediate rewards:
- **Problem**: Agent may optimize wrong objective
- **Solution**: Sparse rewards or careful shaping

## Related Concepts

- [[Reinforcement Learning]] - Rewards in RL context
- [[Deep Q-Network]] - Q-values depend on rewards
- [[Training Process]] - Reward computation in training
- [[Hyperparameters]] - Reward scales in config

## Further Reading

- Ng, A. Y., et al. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML*.

---
*Reward engineering for effective RL training*
