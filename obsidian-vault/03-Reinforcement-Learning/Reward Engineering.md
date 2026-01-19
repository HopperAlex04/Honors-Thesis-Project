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
- **Draw**: `r = 0.0` (neutral, neither good nor bad)

In this project:
- Win: `REWARD_WIN = 1.0`
- Loss: `REWARD_LOSS = -1.0`
- Draw: `REWARD_DRAW = 0.0` (neutral - neither good nor bad)

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
REWARD_DRAW = 0.0             # Terminal: Draw (neutral)
REWARD_INTERMEDIATE = 0.0     # Base for intermediate steps
SCORE_REWARD_SCALE = 0.01     # Scale for score changes
GAP_REWARD_SCALE = 0.005      # Scale for gap changes (half of score scale)
USE_INTERMEDIATE_REWARDS = True  # Must be enabled for score/gap rewards
```

### Design Rationale

1. **Terminal rewards**: Clear signal for game outcome
   - Win: Positive (encourage)
   - Loss: Negative (discourage)
   - Draw: Neutral (neither encourage nor discourage)

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

## Experimental Findings: Sparse vs Intermediate Rewards

### The Sparse Rewards Problem (January 2026)

**Experiment**: Training with `USE_INTERMEDIATE_REWARDS = False` (sparse rewards only)

**Configuration**:
- 20 rounds × 250 episodes = 5,000 total training episodes
- Epsilon decay fixed to reach minimum by Round 1
- Terminal rewards only: Win (+1.0), Loss (-1.0), Draw (0.0)
- No score-based or gap-based intermediate rewards

**Results** (after 7 rounds of training):

| Round | vs Random (1st/2nd) | vs GapMaximizer (1st/2nd) |
|-------|---------------------|---------------------------|
| 0     | 18% / 23%          | 3% / 5%                   |
| 2     | 23% / 27%          | 2% / 11%                  |
| 4     | 21% / 21%          | 3% / 11%                  |
| 6     | 18% / 18%          | 3% / 10%                  |

**Observations**:
1. **No learning progression** - Win rates remained flat at ~20% vs random
2. **Loss decreased normally** - Network was fitting (loss: 0.21 → 0.16)
3. **Agent performed worse than random** - Expected ~50% vs random opponent

**Diagnosis**: The agent learned to minimize TD error but couldn't attribute wins/losses to specific actions. With only terminal rewards, the credit assignment problem was too severe - the agent had no signal about which intermediate actions contributed to the outcome.

### Solution: Enable Intermediate Rewards

**Change**: Set `USE_INTERMEDIATE_REWARDS = True`

**Effect**: Now intermediate steps receive:
```python
reward = REWARD_INTERMEDIATE + (score_change * SCORE_REWARD_SCALE) + (gap_change * GAP_REWARD_SCALE)
```

**Why this helps**:
1. **Credit assignment** - Agent gets immediate feedback when scoring points
2. **Denser signal** - More learning signal per episode (not just at end)
3. **Guides exploration** - Actions that increase score are reinforced immediately

**Preserved Evidence**: The failed experiment is preserved at:
```
experiments/FAILED_experiment_20260119_no_intermediate_rewards/
```

This demonstrates the importance of intermediate rewards for complex games like Cuttle where:
- Episodes can be long (many turns)
- Many actions don't directly cause win/loss
- Credit assignment is difficult with sparse rewards alone

### Key Lesson

For games with long episodes and complex action sequences, **sparse terminal rewards alone are insufficient**. Intermediate rewards (score-based, gap-based) provide the gradient signal needed for effective learning, even though they add design complexity.

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
