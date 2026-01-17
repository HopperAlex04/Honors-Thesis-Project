---
title: Experience Replay
tags: [reinforcement-learning, training, experience-replay]
created: 2026-01-16
related: [Deep Q-Network, Q-Learning, Training Process]
---

# Experience Replay

## Overview

**Experience Replay** is a technique in reinforcement learning where an agent stores past experiences in a buffer and learns from randomly sampled batches instead of learning immediately from each experience. This was introduced with [[Deep Q-Network]] and is crucial for stable, efficient learning.

## Motivation

### Sequential Learning Problem

In standard RL, agent learns from experiences sequentially:
```
(s_t, a_t, r_t, s_{t+1}) → Update immediately → (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}) → ...
```

**Problems**:
1. **Correlation**: Consecutive experiences are highly correlated (same trajectory)
2. **Inefficiency**: Each experience used only once
3. **Instability**: Neural networks can be unstable when learning from correlated data

### Experience Replay Solution

Store experiences, sample randomly:
```
Buffer: {(s, a, r, s', done), ...}
Sample batch randomly → Update network
```

**Benefits**:
- **Decorrelates data**: Random sampling breaks correlation
- **Reuses experiences**: Each experience can be used multiple times
- **Stabilizes learning**: More stable gradient estimates

## How It Works

### 1. Experience Storage

When agent takes action, store experience tuple:
```python
experience = (state, action, reward, next_state, done)
replay_buffer.append(experience)
```

Where:
- **state**: Current observation
- **action**: Action taken
- **reward**: Reward received
- **next_state**: Next observation
- **done**: Whether episode terminated

### 2. Buffer Management

Replay buffer is finite size (e.g., 30,000 experiences):
- **FIFO**: When buffer full, remove oldest experience
- **Circular Buffer**: Overwrite oldest when full
- **Fixed Size**: Maintain constant buffer size

In this project: Buffer size = 30,000 (see [[Hyperparameters]])

### 3. Batch Sampling

When updating network, randomly sample batch:
```python
batch = random.sample(replay_buffer, batch_size)
```

Where batch_size is number of experiences per update (e.g., 128).

### 4. Learning from Batch

For each experience in batch, compute target:
```python
for (s, a, r, s', done) in batch:
    if done:
        target = r
    else:
        target = r + γ * max_{a'} Q(s', a'; θ_target)
    
    loss += (target - Q(s, a; θ))²
```

Update network using batch loss (mean over batch).

## Benefits

### 1. Decorrelation

Random sampling from buffer:
- Mixes experiences from different episodes
- Breaks temporal correlation
- More stable gradient estimates

### 2. Sample Efficiency

Each experience can be used multiple times:
- Same experience appears in different batches
- Better data utilization
- Faster learning (fewer environment interactions needed)

### 3. Stability

Batch updates provide:
- More stable gradient estimates (averaged over batch)
- Reduced variance in updates
- Smoother learning curves

### 4. Diversity

Buffer contains experiences from:
- Different game states
- Different strategies (as agent improves)
- Different outcomes (wins, losses, draws)

This diversity improves generalization.

## Hyperparameters

### Buffer Size

Number of experiences stored:
- **Too small**: Limited diversity, may not contain useful experiences
- **Too large**: Stale experiences (from old policy) may hinder learning
- **Typical**: 10,000 - 1,000,000 depending on problem
- **This project**: 30,000 (see [[Hyperparameters]])

### Batch Size

Number of experiences per update:
- **Too small**: High variance, unstable updates
- **Too large**: Slow updates, memory intensive
- **Typical**: 32 - 256
- **This project**: 128 (see [[Hyperparameters]])

### Sampling Strategy

**Uniform Random Sampling** (used in basic DQN):
- All experiences equally likely
- Simple, effective

**Prioritized Experience Replay** (advanced):
- Sample important experiences more often
- Prioritize experiences with large TD error
- Can improve learning efficiency

This project uses **uniform random sampling**.

## Implementation Considerations

### Memory Efficiency

Store only necessary information:
- States: Can be large (in this project: 468 booleans)
- Actions: Usually small (integer)
- Rewards: Scalar
- Done flags: Boolean

Use efficient data structures (e.g., NumPy arrays, circular buffers).

### Update Frequency

How often to sample batch and update:
- After each action (online): Most responsive, can be slow
- After N actions: Balanced
- After episode: Less frequent, may be less efficient

### Buffer Warm-up

Wait until buffer has minimum experiences before training:
- Prevents learning from too few experiences
- Ensures some diversity in buffer
- **Typical**: Start training after 1,000-10,000 experiences

## Experience Replay in This Project

### Implementation

- **Buffer Size**: 30,000 experiences
- **Batch Size**: 128 experiences per update
- **Sampling**: Uniform random sampling
- **Storage**: Experiences from [[Self-Play]] training games

### Integration with Training

1. Agent plays game ([[Self-Play]] or vs opponents)
2. Experiences collected and stored in buffer
3. Periodically sample batch and update [[Deep Q-Network|DQN]] network
4. Buffer continuously updated with new experiences

See [[Training Process]] for full training loop.

## Limitations

### 1. Memory Requirements

Storing many experiences requires memory:
- Each experience contains states (can be large)
- Large buffers = large memory usage
- **Solution**: Efficient storage, fixed buffer size

### 2. Stale Experiences

Old experiences from outdated policy:
- Agent policy changes, old experiences may be misleading
- **Mitigation**: Fixed-size buffer (FIFO) replaces old experiences

### 3. Off-Policy Learning

Experiences from old policy, learning new policy:
- **This is fine for Q-Learning** (off-policy algorithm)
- May slow learning if policy changes rapidly

## Related Concepts

- [[Deep Q-Network]] - Algorithm that uses experience replay
- [[Q-Learning]] - Off-policy algorithm compatible with replay
- [[Training Process]] - How replay is integrated
- [[Self-Play]] - Source of experiences in this project

## Further Reading

- Lin, L. J. (1992). "Self-improving reactive agents based on reinforcement learning, planning and teaching." *Machine Learning*.
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.

---
*Experience replay technique for stable DQN training*
