# Episode Count and Epsilon Decay Analysis

## Current Configuration

- **Training episodes**: 20 rounds × 250 episodes = **5,000 episodes**
- **Validation episodes**: 250 per round × 20 rounds = **5,000 episodes** (ratio=1.0)
- **Epsilon decay**: `eps_decay = 1500`
- **Epsilon range**: 0.9 → 0.05

## Key Findings

### 1. Epsilon Decays Too Fast

**Problem**: Epsilon reaches minimum (0.05) very early in training.

**Analysis**:
- Total training steps ≈ 34,000 (5000 episodes × ~6-7 steps/episode)
- With `eps_decay = 1500`, epsilon reaches 0.05 by step ~8,500 (round 5)
- For remaining ~25,500 steps, epsilon stays at minimum (0.05)
- This means agent is mostly exploiting (not exploring) for 75% of training

**Epsilon at different stages**:
- Round 0 (~0 steps): ε = 0.90
- Round 5 (~8,500 steps): ε = 0.05 (already at minimum!)
- Round 10 (~17,000 steps): ε = 0.05
- Round 19 (~32,300 steps): ε = 0.05

### 2. Training Convergence

**Embedding Network** (best performer):
- Early rounds (0-4): 61.8% win rate
- Late rounds (15-19): 68.8% win rate
- **Improvement**: +7.0%
- **Status**: Still improving, but rate is slowing

**Boolean Network** (baseline):
- Final win rates: 12-22% (very low)
- Suggests architecture limitations rather than insufficient training

### 3. Episode Count Assessment

**Current: 5,000 training episodes**
- ✅ Sufficient for embedding network (reaching 68-70% win rate)
- ✅ Reasonable training time (~1.5-2 hours per run)
- ⚠️ Boolean network underperforming (likely architecture issue, not episode count)

**Could reduce to 4,000 episodes** if:
- Want faster experiments
- Acceptable to stop slightly earlier
- Still provides 16 rounds × 250 = 4,000 episodes

## Recommendations

### Option 1: Adjust Epsilon Decay (Recommended)

**Increase `eps_decay` to slow down epsilon decay:**

```json
{
  "eps_decay": 11000,
  "eps_decay_comment": "Adjusted for ~34,000 total steps. Epsilon reaches ~0.1 by end of training, allowing gradual exploration-to-exploitation transition."
}
```

**Rationale**:
- With `eps_decay = 11,000`:
  - Round 5: ε ≈ 0.75 (still exploring)
  - Round 10: ε ≈ 0.50 (balanced)
  - Round 15: ε ≈ 0.25 (mostly exploiting)
  - Round 19: ε ≈ 0.10 (near minimum, still some exploration)
- Allows gradual transition from exploration to exploitation
- Better matches the full training duration

### Option 2: Reduce Episodes (Optional)

**If experiments are taking too long:**

```json
{
  "training": {
    "rounds": 16,
    "eps_per_round": 250
  }
}
```

**Total**: 4,000 training episodes
- Still provides good convergence data
- ~20% faster experiments
- Embedding network likely still reaches good performance

### Option 3: Reduce Number of Runs (Selected for Time Efficiency)

**Current**: 7 runs per network type (21 total runs)

**Recommended**: 5 runs per network type (15 total runs)

**Statistical Significance**:
- 5 runs is the **minimum for basic statistical analysis** (per documentation)
- Still allows t-tests and confidence intervals
- 95% CI with n=5: uses t-statistic ≈ 2.776 (vs 2.447 for n=7)
- Slightly wider confidence intervals, but still acceptable for thesis

**Time Savings**:
- Reduces total runs from 21 to 15 (29% reduction)
- Saves ~6 runs worth of time
- Maintains validation quality (keeps validation_episodes_ratio = 1.0)

**Trade-off**:
- Slightly less statistical power than 7 runs
- Still sufficient for thesis-level research
- Acceptable given time constraints

## Recommended Configuration

```json
{
  "eps_decay": 11000,
  "eps_decay_comment": "Adjusted for ~34,000 total training steps. Epsilon decays from 0.9 to ~0.1 over full training, allowing gradual exploration-to-exploitation transition.",
  "training": {
    "rounds": 20,
    "eps_per_round": 250,
    "validation_episodes_ratio": 1.0,
    "validation_opponent": "both"
  }
}
```

**Experiment Configuration**:
- **Runs per type**: 5 (reduced from 7 for time efficiency)
- **Total runs**: 15 (3 types × 5 runs)

**Summary**:
- ✅ Keep 5,000 training episodes (20 rounds × 250)
- ✅ Increase `eps_decay` to 11,000 (better exploration schedule)
- ✅ Keep validation ratio at 1.0 (maintains validation quality)
- ✅ Reduce runs per type to 5 (maintains statistical significance, saves time)
- ✅ Total per run: 5,000 training + 5,000 validation = 10,000 episodes
- ✅ Total experiment: 15 runs (down from 21, saves ~29% time)

## Epsilon Schedule Comparison

| Round | Steps | Current (1500) | Recommended (11000) |
|-------|-------|---------------|---------------------|
| 0     | 0     | 0.90          | 0.90                |
| 5     | 8,500 | **0.05** ⚠️   | 0.75                |
| 10    | 17,000| **0.05** ⚠️   | 0.50                |
| 15    | 25,500| **0.05** ⚠️   | 0.25                |
| 19    | 32,300| **0.05** ⚠️   | 0.10                |

**Current**: Epsilon hits minimum too early, agent stops exploring after round 5.

**Recommended**: Epsilon decays gradually, maintaining exploration throughout training.
