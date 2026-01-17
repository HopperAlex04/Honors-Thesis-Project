# Regression Sources Analysis

This document identifies potential sources of regression in the training code that haven't been accounted for.

## 🔴 CRITICAL ISSUES FOUND

### 1. **Target Network Not Updated on Checkpoint Load** ⚠️ CRITICAL

**Location**: `train_no_features.py` lines 328-344

**Problem**: When loading a checkpoint, the target network is NOT updated to match the policy network. This causes:
- Target network to be out of sync with policy network
- Bellman targets computed with stale Q-values
- Loss increases as policy learns but targets don't update
- Training instability

**Current Code**:
```python
trainee.model.load_state_dict(prev_checkpoint['model_state_dict'])
trainee.policy.load_state_dict(prev_checkpoint['model_state_dict'])
# ❌ MISSING: trainee.target.load_state_dict(...)
```

**Fix**: Always update target network when loading checkpoint:
```python
trainee.target.load_state_dict(prev_checkpoint['model_state_dict'])
```

**Impact**: HIGH - This is likely a major source of regression, especially after checkpoint loading.

---

### 2. **Target Network Update Counter Not Reset on Checkpoint Load** ⚠️ MEDIUM

**Location**: `src/cuttle/players.py` line 610, `train_no_features.py` checkpoint loading

**Problem**: `update_target_counter` is not reset when loading checkpoints. This means:
- Target network might not update at the expected frequency
- If counter was at 499/500, it might update immediately after load
- Or if counter was reset, it might take longer than expected

**Impact**: MEDIUM - Could cause inconsistent target network update timing.

**Fix**: Reset `update_target_counter` when loading checkpoint, or save/restore it.

---

### 3. **Optimizer State Mismatch After Regression Revert** ⚠️ MEDIUM

**Location**: `train_no_features.py` lines 497-510

**Problem**: When reverting due to regression, optimizer is recreated with new LR, but:
- Old optimizer state (momentum, Adam's running averages) is lost
- This is intentional (to prevent momentum from bad training), but could cause:
  - Sudden change in optimization behavior
  - Loss of beneficial momentum from good training

**Current Behavior**: Intentional reset to prevent bad momentum
**Impact**: MEDIUM - This is a trade-off, but worth monitoring.

---

## 🟡 POTENTIAL ISSUES

### 4. **Replay Memory State Inconsistency** ⚠️ LOW-MEDIUM

**Location**: `train_no_features.py` lines 312-320, 472-483

**Problem**: Memory snapshot is taken at start of round, but:
- Memory continues to grow during training
- If regression detected, memory is restored to snapshot
- But `update_target_counter` and other state might be inconsistent

**Impact**: LOW-MEDIUM - Could cause slight inconsistencies, but memory restoration is intentional.

---

### 5. **Epsilon Calculation with Exploration Boost** ⚠️ LOW

**Location**: `src/cuttle/players.py` lines 475-478

**Problem**: `exploration_boost` is applied to epsilon calculation, but:
- If boost is set and not reset properly, epsilon stays high
- Could cause excessive exploration when it should be exploiting

**Current Behavior**: Boost is reset on improvement (line 436)
**Impact**: LOW - Seems properly handled, but worth monitoring.

---

### 6. **Device Mismatch in Optimize** ⚠️ LOW

**Location**: `src/cuttle/players.py` line 558

**Problem**: Device is inferred from policy parameters, but:
- If model and target are on different devices, this could fail
- States from replay memory might be on wrong device

**Current Behavior**: Device is inferred correctly
**Impact**: LOW - Code handles this, but worth verifying in edge cases.

---

### 7. **Learning Rate Decay Timing** ⚠️ LOW

**Location**: `train_no_features.py` lines 346-347

**Problem**: LR decays based on round number, but:
- If training is resumed, rounds might not align with actual training progress
- LR might decay too early or too late

**Impact**: LOW - Minor issue, but could affect training stability.

---

### 8. **Replay Memory Sampling Strategy** ⚠️ LOW

**Location**: `src/cuttle/players.py` lines 657-698

**Problem**: `mix_old_new` sampling strategy:
- 50% recent, 50% old experiences
- This is intentional to prevent forgetting, but:
  - Could slow learning of new strategies
  - Old experiences might be from very different skill level

**Impact**: LOW - This is a design choice, not necessarily a bug.

---

## ✅ ALREADY ACCOUNTED FOR

1. ✅ Q-value clipping - Fixed (only clips extreme outliers)
2. ✅ Gradient clipping - Configured (5.0)
3. ✅ Target network updates - Working (every 500 steps)
4. ✅ Bellman equation - Verified correct
5. ✅ Loss calculation - Using Huber loss (robust)
6. ✅ Replay memory restoration - Implemented on regression
7. ✅ Optimizer state reset - Intentional on regression
8. ✅ Grace rounds - Just added

---

## 🔧 RECOMMENDED FIXES

### Priority 1: Fix Target Network on Checkpoint Load

```python
# In train_no_features.py, after loading checkpoint:
trainee.model.load_state_dict(prev_checkpoint['model_state_dict'])
trainee.policy.load_state_dict(prev_checkpoint['model_state_dict'])
trainee.target.load_state_dict(prev_checkpoint['model_state_dict'])  # ADD THIS
trainee.target.eval()  # Ensure target stays in eval mode
```

### Priority 2: Save/Restore Target Update Counter

```python
# Save in checkpoint:
'target_update_counter': trainee.update_target_counter

# Restore when loading:
if 'target_update_counter' in prev_checkpoint:
    trainee.update_target_counter = prev_checkpoint['target_update_counter']
```

### Priority 3: Add Validation for Target Network Sync

Add a check to verify target network matches policy after checkpoint load:
```python
# Verify target network is synced
policy_params = list(trainee.policy.parameters())
target_params = list(trainee.target.parameters())
for p, t in zip(policy_params, target_params):
    if not torch.allclose(p.data, t.data):
        print("WARNING: Target network not synced with policy!")
        break
```

---

## 📊 SUMMARY

**Critical Issues**: 1 (target network not updated)
**Medium Issues**: 2 (counter reset, optimizer state)
**Low Issues**: 5 (various edge cases)

**Most Likely Source of Regression**: Target network not being updated on checkpoint load. This would cause immediate divergence after each round starts.

