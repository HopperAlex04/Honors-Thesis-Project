---
title: Training Schedule and Rounds
tags: [training, hyperparameters, schedule, rounds]
created: 2026-01-16
related: [Hyperparameters, Self-Play, Statistical Significance and Multiple Runs]
---

# Training Schedule and Rounds

## Overview

Training is organized into **rounds**, where each round consists of:
1. **Self-play training**: Agent plays against itself for a fixed number of episodes
2. **Checkpoint saving**: Model state saved for next round
3. **Validation**: Agent tested against fixed opponents to track progress

The key questions are:
- **How long should each round be?** (episodes per round)
- **How many rounds are needed?** (total rounds)

## Current Configuration

### Quick Test Mode (Current)
```json
{
  "rounds": 4,
  "eps_per_round": 500,
  "quick_test_mode": true
}
```
- **Total episodes**: 2,000
- **Purpose**: Fast hyperparameter testing

### Full Training Mode (Default)
```json
{
  "rounds": 10,
  "eps_per_round": 500,
  "quick_test_mode": false
}
```
- **Total episodes**: 5,000
- **Purpose**: Full training run

## Round Structure

### What Happens in Each Round

1. **Load Checkpoint**: Load model from previous round
2. **Self-Play Training**: `eps_per_round` episodes of self-play
3. **Save Checkpoint**: Save model state for next round
4. **Validation**: Test against fixed opponents
   - Validation episodes = `eps_per_round × validation_episodes_ratio` (default: 0.5)
   - Example: 500 episodes → 250 validation episodes

### Round Length Considerations

#### Minimum Episode Count

**Recommendation: 200-500 episodes per round**

**Rationale**:
- **Early stopping**: Requires at least 200 episodes before checking (`min_episodes: 200`)
- **Stable learning**: Need enough episodes for meaningful updates
- **Validation accuracy**: More episodes = more reliable win rate estimates
  - 200 episodes: ~3.5% standard error
  - 500 episodes: ~2.2% standard error

#### Too Few Episodes (< 100)

**Problems**:
- High variance in validation metrics
- Insufficient data for stable learning
- Early stopping may trigger prematurely
- Overhead dominates (checkpointing, validation)

#### Too Many Episodes (> 1000)

**Problems**:
- Less frequent progress tracking
- Fewer checkpoints for analysis
- Slower feedback on training issues
- More work lost if training fails

## Reduction of Rounds (Why We Use Fewer Rounds)

### Decision

After analyzing previous experiments (architecture comparison and game-based scaling screening), we **reduced the number of rounds** used for screening and often for full runs. The default or recommended configuration moved from **20 rounds** to **10 rounds** (or similar) where appropriate.

### Why the Reduction Was Made

1. **Peak performance occurs mid-training, not at the final round.**  
   Validation win rate vs GapMaximizer typically **peaks between rounds 5 and 15**, then often **declines** by round 19. Examples:
   - **large_hidden_embedding:** Peak at round 7 (63.2%), final round 19 = 52.8%.
   - **linear_embedding:** Peak at round 15 (60.8%), final 58.4%.
   - **game_based_embedding (scale 11):** Peak at round 10 (60.8%), final 49.6%.

2. **Win rate plateaus by round 5–10.**  
   For runs that learn, performance reaches ~95% of its “final” level by round 5–10. Rounds 11–19 add little improvement and, in many runs, coincide with a **decline** in validation win rate (overfitting, instability, or epsilon/exploration effects).

3. **Extra rounds waste time and can worsen reported results.**  
   Running 20 rounds doubles training time compared to 10 rounds, while the second half often does not improve—and sometimes hurts—the validation metric we care about. Reporting “round 19” can understate the best performance achieved.

4. **Screening experiments need speed, not maximum length.**  
   For scaling and architecture screening, the goal is to compare configs and find promising scales. Capturing performance by round 10 is sufficient; running to 20 rounds is unnecessary and costly.

### Implications

- **Screening:** Use **10 rounds** (or at most 12). Total episodes: 2,500–3,000 per run instead of 5,000. Roughly **50% less time per run** with no evidence of losing useful signal and some evidence of avoiding post-peak degradation.
- **Final evaluation:** Prefer **best checkpoint by validation win rate** (best round) rather than always reporting the last round. Consider **early stopping** when validation win rate has not improved (or has dropped) for several rounds.
- **Full training (thesis):** If keeping a longer schedule, pair it with best-checkpoint selection or early stopping so that “final” results reflect the best validation performance, not the last round.

Detailed analysis and tables: see `docs/ROUNDS_ANALYSIS.md` in the project repo.

---

## Number of Rounds

### Total Training Episodes

**For thesis experiments: 5,000-10,000 total episodes**

**Rationale**:
- **Epsilon decay**: Current `eps_decay: 28510` suggests ~28K episodes for full decay
- **Convergence**: Most RL agents need 5K-10K episodes for reasonable performance
- **Multiple runs**: With 7 runs per network type, need reasonable per-run time

### Recommended Configurations

#### Option 1: Moderate Granularity (Recommended)

```json
{
  "rounds": 20,
  "eps_per_round": 250,
  "validation_episodes_ratio": 0.5
}
```

**Total episodes**: 5,000
**Validation episodes per round**: 125

**Benefits**:
- ✅ Good progress tracking (20 data points)
- ✅ Reasonable validation sample size (125 episodes)
- ✅ Manageable overhead (~2-3% total time)
- ✅ Enough episodes per round for stable learning (250)

**Trade-offs**:
- More frequent checkpoints (good for recovery)
- More validation data points (better for analysis)

#### Option 2: Fine Granularity (For Detailed Analysis)

```json
{
  "rounds": 25,
  "eps_per_round": 200,
  "validation_episodes_ratio": 0.5
}
```

**Total episodes**: 5,000
**Validation episodes per round**: 100

**Benefits**:
- ✅ Very fine progress tracking (25 data points)
- ✅ More checkpoints for detailed analysis
- ✅ Better for identifying when improvements occur

**Trade-offs**:
- Slightly higher validation variance (100 episodes)
- More overhead (but still manageable)

#### Option 3: Coarse Granularity (Faster Training)

```json
{
  "rounds": 10,
  "eps_per_round": 500,
  "validation_episodes_ratio": 0.5
}
```

**Total episodes**: 5,000
**Validation episodes per round**: 250

**Benefits**:
- ✅ Lower overhead (fewer checkpoints)
- ✅ Very stable validation estimates (250 episodes)
- ✅ Less frequent interruptions

**Trade-offs**:
- Fewer progress tracking points (10 data points)
- Less granular analysis
- More work lost if training fails mid-round

## Recommendations by Use Case

### For Full Thesis Experiments

**Recommended**: **20 rounds × 250 episodes = 5,000 total episodes**

**Rationale**:
- Good balance of tracking and efficiency
- Enough validation episodes for reliable estimates
- Reasonable number of checkpoints
- Works well with learning rate decay (every 5 rounds = 4 decays)

### For Quick Testing

**Recommended**: **5 rounds × 100 episodes = 500 total episodes**

```json
{
  "quick_test_mode": true,
  "quick_test_rounds": 5,
  "quick_test_eps_per_round": 100
}
```

**Rationale**:
- Fast iteration for hyperparameter tuning
- Enough rounds to see trends
- Minimal time investment

### For Extended Training

**Recommended**: **25 rounds × 400 episodes = 10,000 total episodes**

**Rationale**:
- More training for better convergence
- Still good tracking (25 data points)
- Enough episodes per round (400) for stable learning

## Learning Rate Decay Considerations

### Current Configuration

```json
{
  "lr_decay_interval": 5,
  "lr_decay_rate": 0.9
}
```

**Learning rate decays every 5 rounds**

### With Different Round Counts

| Rounds | LR Decays | Episodes Between Decays |
|--------|-----------|-------------------------|
| 10 | 2 | 2,500 episodes |
| 20 | 4 | 1,250 episodes |
| 25 | 5 | 1,000 episodes |

**Recommendation**: Keep `lr_decay_interval: 5` for 20-25 rounds
- Provides 4-5 decays over training
- Reasonable decay frequency
- Not too aggressive

## Epsilon Decay Considerations

### Current Configuration

```json
{
  "eps_start": 0.90,
  "eps_end": 0.05,
  "eps_decay": 28510
}
```

**Epsilon decays over 28,510 episodes**

### With 5,000 Total Episodes

- Epsilon at end: `0.90 - (0.90 - 0.05) × (5000 / 28510) ≈ 0.75`
- Still high exploration at end of training

**Options**:

1. **Keep current**: High exploration throughout (good for exploration)
2. **Reduce eps_decay**: Faster decay (e.g., 10000 → epsilon ≈ 0.52 at end)
3. **Increase total episodes**: Train longer to reach lower epsilon

**Recommendation**: For 5,000 episodes, consider `eps_decay: 10000-15000` to reach epsilon ≈ 0.3-0.5 by end

## Validation Considerations

### Validation Sample Size

**Current**: `validation_episodes_ratio: 0.5`

With 250 episodes per round:
- **Validation episodes**: 125 per round
- **Standard error**: ~4.4% (acceptable)
- **95% CI width**: ~8.8%

**Recommendation**: Keep 0.5 ratio for 200-500 episodes per round

### Validation Frequency

Validation happens **every round**, which is good for:
- Tracking progress over time
- Detecting regressions early
- Building training curves

## Computational Considerations

### Time Per Round

Assuming ~2-5 seconds per episode:
- **250 episodes**: ~10-20 minutes per round
- **500 episodes**: ~20-40 minutes per round

### Total Training Time

With 20 rounds × 250 episodes:
- **Training time**: ~3-7 hours per run
- **With 7 runs**: ~21-49 hours total per network type
- **With 3 network types**: ~63-147 hours total (~2.5-6 days)

### Overhead

**Checkpointing + Validation overhead**: ~2-3% of total time
- More rounds = slightly more overhead
- But benefits (tracking, recovery) outweigh costs

## Best Practices

### 1. Start with Moderate Configuration

Begin with **20 rounds × 250 episodes**:
- Good balance
- Can adjust based on results
- Reasonable training time

### 2. Monitor Progress

Watch for:
- **Plateauing**: Win rate stops improving
- **Divergence**: Loss increases, performance decreases
- **Convergence**: Stable performance over multiple rounds

### 3. Adjust Based on Results

**If converging quickly**:
- Can stop early (use early stopping)
- Or reduce total episodes

**If not converging**:
- Increase total episodes (more rounds or more episodes per round)
- Check hyperparameters

### 4. Use Early Stopping

Current early stopping:
- Checks every 50 episodes
- Needs at least 200 episodes
- Can stop if loss diverges

**Works well with 200+ episodes per round**

## Recommended Configuration for Thesis

### Full Training (Long Schedule)

```json
{
  "training": {
    "rounds": 20,
    "eps_per_round": 250,
    "quick_test_mode": false,
    "validation_episodes_ratio": 0.5,
    "validation_opponent": "both"
  },
  "eps_decay": 12000,
  "lr_decay_interval": 5
}
```

**Total**: 5,000 episodes, 20 validation checkpoints. Prefer **best checkpoint by validation win rate** over “final round” when reporting (see [[#Reduction of Rounds (Why We Use Fewer Rounds)]]).

### Screening / Shorter Runs (Recommended for Scaling and Architecture Screening)

```json
{
  "training": {
    "rounds": 10,
    "eps_per_round": 250,
    "quick_test_mode": false,
    "validation_episodes_ratio": 0.5,
    "validation_opponent": "both"
  }
}
```

**Total**: 2,500 episodes, 10 validation checkpoints. Based on rounds analysis: performance typically plateaus or peaks by round 10; extra rounds often degrade validation win rate and double run time. See [[#Reduction of Rounds (Why We Use Fewer Rounds)]].

### Quick Testing

```json
{
  "training": {
    "rounds": 5,
    "eps_per_round": 100,
    "quick_test_mode": true,
    "quick_test_rounds": 5,
    "quick_test_eps_per_round": 100
  }
}
```

**Total**: 500 episodes, 5 validation checkpoints

## Related Concepts

- [[Hyperparameters]] - Full hyperparameter configuration
- [[Self-Play]] - Training methodology
- [[Statistical Significance and Multiple Runs]] - Running multiple experiments
- [[Early Stopping]] - Stopping training early if needed

## References

- **Rounds analysis** (why we reduced rounds): `docs/ROUNDS_ANALYSIS.md` in the project repository. Summarizes plateau/peak-by-round and post-peak decline from architecture comparison and scaling experiments.

---
*Guidelines for configuring training rounds and schedule*
