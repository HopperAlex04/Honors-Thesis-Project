---
title: Training Time Estimates
tags: [training, time, estimates, computational-resources]
created: 2026-01-16
related: [Training Schedule and Rounds, Statistical Significance and Multiple Runs]
---

# Training Time Estimates

## Overview

This document provides time estimates for training runs based on the 20 rounds × 250 episodes configuration.

**Note**: These estimates are based on actual test run measurements (January 2026). A test run with 20 training episodes showed:
- Training episodes: **0.26 seconds average** (range: 0.05-0.73s)
- Validation episodes: **0.46 seconds average** (range: 0.08-0.71s)
- Much faster than initial theoretical estimates!

## Configuration

**Training Schedule**:
- **Rounds**: 20
- **Episodes per round**: 250
- **Total training episodes**: 5,000
- **Validation episodes per round**: 125 (50% ratio)
- **Total validation episodes**: 2,500
- **Total episodes (training + validation)**: 7,500

## Time Per Episode Estimates

### Actual Measured Times (Test Run)

**Test Configuration**: 1 round, 20 training episodes, 4 validation episodes

**Training Episodes (Self-Play)**:
- **Average**: 0.26 seconds per episode
- **Median**: 0.24 seconds per episode
- **Range**: 0.05 - 0.73 seconds
- **Total for 20 episodes**: 5.15 seconds

**Validation Episodes**:
- **Average**: 0.46 seconds per episode
- **Range**: 0.08 - 0.71 seconds
- **Total for 4 episodes**: 1.83 seconds

**Note**: Validation is actually slightly slower than training in this test, likely due to logging overhead and game length variation.

### Training Episodes (Self-Play)

**Measured time per episode**: **0.2-0.3 seconds** (average 0.26s)

**Breakdown**:
- Game execution: 0.1-0.2 seconds (fast games with early terminations)
- Neural network forward pass: 0.05-0.1 seconds (CPU-based, efficient)
- Experience replay sampling: 0.01-0.02 seconds
- Optimization step: 0.02-0.05 seconds (batch processing)
- Logging and overhead: 0.01-0.02 seconds

**Factors affecting time**:
- **Game length**: Longer games = more turns = more time (0.05s for quick wins, 0.7s for long games)
- **Network size**: Larger networks = slightly slower forward passes
- **CPU performance**: Single-threaded performance matters
- **Replay buffer size**: Larger buffers = slightly slower sampling

### Validation Episodes

**Measured time per episode**: **0.3-0.5 seconds** (average 0.46s)

**Breakdown**:
- Game execution: 0.2-0.4 seconds
- Neural network forward pass: 0.05-0.1 seconds (greedy, no exploration)
- No optimization: Faster than training
- Logging: 0.01-0.02 seconds

**Note**: Validation can be similar or slightly slower than training due to:
- More complete games (less early termination)
- Full logging enabled
- Both positions tested

## Time Per Round

### Training Phase

**250 training episodes** (based on measured 0.26s average):
- **Conservative estimate**: 250 × 0.2s = **50 seconds** (~0.8 minutes)
- **Average estimate**: 250 × 0.26s = **65 seconds** (~1.1 minutes)
- **Pessimistic estimate**: 250 × 0.4s = **100 seconds** (~1.7 minutes)

### Validation Phase

**125 validation episodes** (per opponent, based on measured 0.46s average):
- **Conservative estimate**: 125 × 0.3s = **37.5 seconds** (~0.6 minutes)
- **Average estimate**: 125 × 0.46s = **57.5 seconds** (~1.0 minutes)
- **Pessimistic estimate**: 125 × 0.6s = **75 seconds** (~1.3 minutes)

**With 2 opponents** (randomized + gapmaximizer):
- **Conservative**: 2 × 37.5s = **75 seconds** (~1.3 minutes)
- **Average**: 2 × 57.5s = **115 seconds** (~1.9 minutes)
- **Pessimistic**: 2 × 75s = **150 seconds** (~2.5 minutes)

### Round Overhead

**Checkpointing and setup**:
- Load checkpoint: ~0.5-1 seconds
- Save checkpoint: ~0.5-1 seconds
- Learning rate decay (every 5 rounds): ~0.1 seconds
- **Total overhead per round**: ~1-2 seconds

### Total Time Per Round

**Conservative estimate**:
- Training: 50s
- Validation: 75s
- Overhead: 1s
- **Total**: ~126 seconds (~2.1 minutes)

**Average estimate**:
- Training: 65s
- Validation: 115s
- Overhead: 1.5s
- **Total**: ~181.5 seconds (~3.0 minutes)

**Pessimistic estimate**:
- Training: 100s
- Validation: 150s
- Overhead: 2s
- **Total**: ~252 seconds (~4.2 minutes)

## Time Per Full Run (One Network Type)

**20 rounds total**:

### Conservative Estimate
- 20 rounds × 2.1 minutes = **42 minutes** (~0.7 hours)

### Average Estimate
- 20 rounds × 3.0 minutes = **60 minutes** (~1.0 hours)

### Pessimistic Estimate
- 20 rounds × 4.2 minutes = **84 minutes** (~1.4 hours)

## Time for All Three Network Types

**Three network types**: Boolean, Embedding, Multi-Encoder

### Sequential Execution (One After Another)

**Conservative**:
- 3 × 0.7 hours = **2.1 hours**

**Average**:
- 3 × 1.0 hours = **3.0 hours**

**Pessimistic**:
- 3 × 1.4 hours = **4.2 hours**

### Parallel Execution (If Resources Allow)

If you can run 3 network types in parallel:

**Conservative**:
- **0.7 hours** (same as single run)

**Average**:
- **1.0 hours** (same as single run)

**Pessimistic**:
- **1.4 hours** (same as single run)

**Note**: Requires 3× CPU/memory resources

## Realistic Estimate

**Recommended estimate**: **Average case** (~1 hour per network type)

**Rationale**:
- Based on actual measured times (0.26s per training episode)
- Most episodes are very fast (0.2-0.3 seconds)
- Some longer games (up to 0.7s) but rare
- Validation adds ~1-2 minutes per round
- Overhead is minimal (~1-2 seconds per round)

**Total for all three networks (sequential)**: **~3 hours**

## Breakdown by Component

### Per Network Type (5,000 training + 2,500 validation episodes)

**Training time** (5,000 episodes, based on measured 0.26s average):
- Conservative: 5,000 × 0.2s = 1,000s (~16.7 minutes)
- Average: 5,000 × 0.26s = 1,300s (~21.7 minutes)
- Pessimistic: 5,000 × 0.4s = 2,000s (~33.3 minutes)

**Validation time** (2,500 episodes, based on measured 0.46s average):
- Conservative: 2,500 × 0.3s = 750s (~12.5 minutes)
- Average: 2,500 × 0.46s = 1,150s (~19.2 minutes)
- Pessimistic: 2,500 × 0.6s = 1,500s (~25.0 minutes)

**Overhead** (20 rounds):
- Checkpointing: 20 × 1s = 20s (~0.3 minutes)
- Learning rate decay: 4 × 0.1s = 0.4s (negligible)
- **Total overhead**: ~20-30 seconds

## Factors That Can Affect Time

### Faster Training

**If episodes are shorter** (1-2 seconds):
- Games end quickly (early wins)
- Fewer turns per game
- **Could reduce time by 30-40%**

### Slower Training

**If episodes are longer** (5-8 seconds):
- Longer games (more turns)
- More complex game states
- **Could increase time by 50-100%**

### System Performance

**CPU-bound training**:
- Single-threaded performance matters
- More CPU cores don't help (unless parallelizing runs)
- CPU speed directly affects episode time

**Memory**:
- Replay buffer size affects sampling time
- Larger networks use more memory
- Should not be a bottleneck with current sizes

## Recommendations

### Planning

**Use average estimate**: **~1 hour per network type**

**For scheduling**:
- **Single network**: Plan for 1-1.5 hours (with buffer)
- **All three networks (sequential)**: Plan for 3-4 hours
- **All three networks (parallel)**: Plan for 1-1.5 hours (if resources allow)

### Monitoring

**Track actual times**:
- First round will give you real episode times
- Adjust estimates based on actual performance
- Use `episode_elapsed_time` from training logs

### Optimization

**If training is too slow**:
- Reduce episodes per round (but maintain minimum 200)
- Reduce validation episodes ratio (but maintain reliability)
- Optimize code (if bottlenecks found)

**If training is faster than expected**:
- Can increase episodes per round
- Can increase total rounds
- Can add more validation

## Example Timeline

### Sequential Execution

**All three networks in one session**:
- Start: 9:00 AM
- Boolean network: Complete by 10:00 AM (1 hour)
- Embedding network: Complete by 11:00 AM (1 hour)
- Multi-Encoder network: Complete by 12:00 PM (1 hour)
- **Total**: 3 hours for all three networks

### Alternative: Parallel Execution

**If running in parallel** (3 systems or sufficient resources):
- Start all three: 9:00 AM
- Estimated completion: 10:00 AM (1 hour)
- **Total**: 1 hour for all three networks

## Storage Considerations

**Per network type**:
- Models: ~50-100 MB (20 checkpoints)
- Logs: ~500 MB - 1 GB (metrics, actions)
- **Total per network**: ~1-2 GB

**All three networks**: ~3-6 GB

**For 7 runs per network type** (statistical significance):
- **Per network type**: ~7-14 GB
- **All three networks**: ~21-42 GB
- **Total for all runs**: ~147-294 GB (21 runs total)

## Related Concepts

- [[Training Schedule and Rounds]] - Round configuration
- [[Statistical Significance and Multiple Runs]] - Multiple runs needed
- [[Hyperparameters]] - Configuration affecting training time

---
*Time estimates for training runs*
