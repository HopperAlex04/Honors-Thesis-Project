# Analysis: More Rounds vs. More Episodes Per Round

## Current Setup
- **Rounds**: 10
- **Episodes per round**: 500
- **Total episodes**: 5,000
- **Checkpoint frequency**: Every 500 episodes
- **Validation frequency**: Every 500 episodes
- **LR decay**: Every 3 rounds (1,500 episodes)

## Option: More Rounds, Fewer Episodes

### Example Configurations:

**Option A: Moderate Change**
- Rounds: 20
- Episodes per round: 250
- Total episodes: 5,000 (same)
- Checkpoint frequency: Every 250 episodes
- Validation frequency: Every 250 episodes
- LR decay: Every 3 rounds (750 episodes)

**Option B: Aggressive Change**
- Rounds: 50
- Episodes per round: 100
- Total episodes: 5,000 (same)
- Checkpoint frequency: Every 100 episodes
- Validation frequency: Every 100 episodes
- LR decay: Every 3 rounds (300 episodes)

## Benefits of More Rounds

### 1. **Better Progress Tracking** ✅
- More data points for win rate graphs
- Finer granularity in seeing improvements
- Easier to identify when learning plateaus
- Better for research documentation

### 2. **More Frequent Validation** ✅
- Catch performance issues earlier
- More opportunities to detect regressions
- Better understanding of agent's current capability
- More validation data for analysis

### 3. **Better Recovery from Issues** ✅
- More frequent checkpoints = less work lost
- Can revert to earlier checkpoints more easily
- Better for experimentation (can stop/restart more flexibly)

### 4. **More Frequent Opponent Updates (Self-Play)** ✅
- Agent plays against newer versions of itself more often
- Could help with non-stationarity (opponent changes more frequently)
- More "fresh" training data

### 5. **Better for Research** ✅
- More granular analysis of training dynamics
- More checkpoints to compare
- Better for ablation studies
- More data points for statistical analysis

## Drawbacks of More Rounds

### 1. **More Overhead** ⚠️
- More time spent on checkpointing/loading
- More time spent on validation
- More disk I/O operations
- **Impact**: ~1-2% overhead per round (checkpoint save/load + validation)

### 2. **Less Stable Training** ⚠️
- More frequent interruptions
- Less continuous learning
- More opportunities for numerical issues
- **Impact**: Minimal if implemented correctly

### 3. **Validation Sample Size** ⚠️
- Fewer episodes per validation = higher variance
- Win rate estimates less reliable with small samples
- **Impact**: With 100 episodes, standard error is ~5% (vs ~2.2% with 500)

### 4. **Self-Play Stability** ⚠️
- Opponent changes more frequently
- Less time to adapt to each opponent version
- Could lead to more unstable training
- **Impact**: Unclear - could be good (faster adaptation) or bad (less stable)

### 5. **Learning Rate Decay Timing** ⚠️
- LR decays every 3 rounds
- With 50 rounds, LR decays 16 times (vs 3 times with 10 rounds)
- Might decay too frequently
- **Solution**: Adjust `lr_decay_interval` based on rounds

## Recommendations

### For Your Research: **YES, Use More Rounds**

**Recommended Configuration:**
```json
{
  "training": {
    "rounds": 25,
    "eps_per_round": 200,
    "quick_test_mode": false,
    "quick_test_rounds": 5,
    "quick_test_eps_per_round": 50
  }
}
```

**Total episodes: 5,000** (same as before)

### Why This Works Well:

1. **200 episodes per round**:
   - Still enough for stable win rate estimates (~3.5% standard error)
   - Enough for meaningful learning per round
   - Not too frequent (overhead is manageable)

2. **25 rounds**:
   - Good granularity for tracking progress
   - Enough checkpoints for analysis
   - LR decays every 3 rounds = 8 decays total (reasonable)
   - More validation data points

3. **Better for Research**:
   - More data points for graphs
   - More checkpoints to analyze
   - Better tracking of training dynamics
   - Easier to identify when improvements happen

### Adjustments Needed:

1. **Update LR decay interval** (if using rounds-based):
   ```json
   "lr_decay_interval": 5  // Decay every 5 rounds instead of 3
   ```

2. **Update regression detection** (if using rounds):
   - Current: `REGRESSION_WINDOW = 3` rounds
   - With 25 rounds, this is still reasonable
   - Consider: `REGRESSION_WINDOW = 5` for more stability

3. **Validation sample size**:
   - 200 episodes per validation is still good
   - Standard error: ~3.5% (acceptable)
   - Can increase validation episodes if needed

## Alternative: Hybrid Approach

**For Quick Testing:**
- Fewer rounds, more episodes (faster, less overhead)

**For Full Training:**
- More rounds, fewer episodes (better tracking, more data)

**Configuration:**
```json
{
  "training": {
    "rounds": 25,
    "eps_per_round": 200,
    "quick_test_mode": true,
    "quick_test_rounds": 5,
    "quick_test_eps_per_round": 100  // Fewer rounds for quick tests
  }
}
```

## Conclusion

**Yes, it's better to use more rounds with fewer episodes per round** for research purposes, especially with the fixed win rate tracking. The benefits (better tracking, more data points, better recovery) outweigh the minor overhead costs.

**Recommended**: 20-25 rounds × 200-250 episodes = 4,000-5,000 total episodes

This gives you:
- ✅ Better progress tracking
- ✅ More validation data
- ✅ More checkpoints for analysis
- ✅ Better recovery from issues
- ✅ Still stable training (200+ episodes per round)
- ✅ Manageable overhead (~2-3% total time)

