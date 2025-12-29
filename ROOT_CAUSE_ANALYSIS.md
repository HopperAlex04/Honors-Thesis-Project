# Root Cause Analysis: Rising Loss in Self-Play

## The Problem
Loss consistently rises over training despite:
- ✅ Target network updating correctly
- ✅ Bellman equation calculated correctly  
- ✅ Hyperparameters adjusted (LR, clipping, etc.)
- ✅ All mechanics verified in debug script

## Root Cause: Self-Play Non-Stationarity

In self-play, **the environment is non-stationary** - the opponent (which is yourself) keeps improving. This creates a fundamental problem:

### The Issue

1. **Distribution Shift**: As the agent improves, it generates different states and strategies
2. **Replay Memory Contamination**: Old experiences from when the agent was worse become less relevant
3. **Moving Target**: The network tries to learn from a mix of:
   - Old experiences (when agent was worse, easier to predict)
   - New experiences (when agent is better, harder to predict)
4. **Incompatible Data**: The network can't satisfy both old and new experiences simultaneously
5. **Result**: Loss increases as the network tries to fit incompatible data

### Why This Happens

```
Episode 100: Agent plays poorly → Easy states → Low Q-values
Episode 500: Agent plays better → Hard states → Higher Q-values
Episode 1000: Agent plays well → Complex states → Very high Q-values

Replay memory contains ALL of these:
- Old: Q(s,a) ≈ 2.0 (easy game)
- New: Q(s,a) ≈ 8.0 (hard game)

Network tries to learn: Q(s,a) = ???
- If it learns 2.0 → Wrong for new states → High loss on new data
- If it learns 8.0 → Wrong for old states → High loss on old data
- Can't satisfy both → Loss keeps rising
```

## Why Loss Rising Doesn't Mean Failure

**Important**: In self-play, **rising loss can be normal** if:
- ✅ Win rates are improving
- ✅ Agent is learning new strategies
- ✅ Performance against fixed opponents is improving

The loss metric becomes less meaningful because:
- The data distribution keeps changing
- Old experiences become "outdated"
- The network is learning harder, more complex strategies

## Solutions

### 1. **Accept Rising Loss** (If Performance Improves)
- Monitor win rates, not just loss
- If agent is improving, loss rising is expected
- This is normal in self-play scenarios

### 2. **Reduce Replay Buffer Size**
- Smaller buffer = faster aging of old experiences
- Current: 100,000 (might be too large)
- Try: 50,000 or even 25,000

### 3. **Prioritized Experience Replay**
- Focus learning on important/recent experiences
- Reduces impact of outdated experiences

### 4. **Periodic Memory Clearing**
- Clear old experiences periodically
- Keep only recent, relevant experiences

### 5. **Separate Training Phases**
- Phase 1: Learn basics (small buffer, clear often)
- Phase 2: Refine strategies (larger buffer, keep recent)

## Verification

To confirm this is the issue, check:
1. **Win rates**: Are they improving despite rising loss?
2. **Loss vs Performance**: Plot loss vs win rate - if win rate improves while loss rises, it's non-stationarity
3. **Recent vs Old Experiences**: Compare loss on recent vs old experiences in replay memory

## Recommendation

**If win rates are improving**: The rising loss is expected and not a problem. Focus on performance metrics instead.

**If win rates are NOT improving**: Then there's a real training issue that needs fixing.

