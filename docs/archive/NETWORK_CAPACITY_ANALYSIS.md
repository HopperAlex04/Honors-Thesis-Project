# Network Capacity Analysis: Can Architecture Limit Win Rate?

## Current Network Architecture

**Architecture**: `480 → 256 → 128 → 3,157`

**Parameters**: 564,153 total
- Embedding: 864 (vocab_size=54, dim=16)
- Linear Layers: 563,289
  - Layer 1 (480→256): 123,136 params (22%)
  - Layer 2 (256→128): 32,896 params (6%)
  - Layer 3 (128→3157): 407,253 params (72%) ⚠️ **BOTTLENECK**

**Input Dimension**: 480 features
- Zone arrays (hand, field, deck, scrap, etc.)
- Embedded stack (5 × 16 = 80)
- Embedded effect_shown (2 × 16 = 32)
- Optional features (scores, highest point values)

**Output Dimension**: 3,157 actions (very large action space!)

---

## Capacity Assessment

### ✅ **Network Size is Reasonable**
- **564K parameters** is in the "medium" range for DQN
- Comparable to successful Atari DQN implementations
- Should be sufficient for most card game strategies

### ⚠️ **Potential Bottlenecks**

#### 1. **Large Action Space Compression** (HIGH CONCERN)
- Final layer: **128 → 3,157** (72% of all parameters)
- Only 128 hidden units to distinguish between 3,157 actions
- **Compression ratio**: 24.6:1 (very high!)
- This could limit the network's ability to learn nuanced action distinctions

**Impact**: The network might struggle to:
- Distinguish between similar actions
- Learn complex action sequences
- Develop sophisticated strategies requiring precise action selection

#### 2. **Narrow Hidden Layers** (MEDIUM CONCERN)
- Hidden layers: 256 → 128 (shrinking)
- Only 128 units in the final hidden layer
- May not have enough capacity to represent complex game state → action mappings

**Impact**: Limited representational capacity for:
- Complex strategic patterns
- Long-term planning
- Multi-step tactical sequences

#### 3. **Shallow Architecture** (LOW CONCERN)
- Only 2 hidden layers
- Deep networks can learn hierarchical features
- But for card games, 2 layers is often sufficient

---

## Could This Limit Win Rate?

### **YES, if:**

1. **Action Discrimination is Critical**
   - If the game requires distinguishing between many similar actions
   - If optimal play requires precise action selection
   - Current: 128 units for 3,157 actions = very compressed representation

2. **Complex Strategies Require More Capacity**
   - If advanced strategies need more hidden units
   - If the network is saturated (all capacity used)
   - Current: 256→128 might be too narrow for complex patterns

3. **Plateau Indicates Capacity Limit**
   - If win rate plateaus despite continued training
   - If loss decreases but win rate doesn't improve
   - This suggests the network can't learn more complex strategies

### **NO, if:**

1. **Other Factors Are Limiting**
   - Training instability (loss rising)
   - Exploration issues (not seeing good strategies)
   - Reward shaping problems
   - Self-play non-stationarity

2. **Network Has Unused Capacity**
   - If loss is still decreasing
   - If network isn't overfitting
   - If training is still making progress

---

## Recommendations

### **Option 1: Increase Hidden Layer Width** (RECOMMENDED FIRST)
```python
# Current: 480 → 256 → 128 → 3157
# Proposed: 480 → 512 → 256 → 3157
# Or:       480 → 256 → 256 → 3157
```

**Benefits**:
- More capacity in hidden layers
- Better feature representation
- Still manageable parameter count (~1-2M)

**Trade-offs**:
- Slightly slower training
- More memory usage
- Might overfit if too large

### **Option 2: Add Another Hidden Layer** (IF OPTION 1 DOESN'T HELP)
```python
# Proposed: 480 → 512 → 256 → 128 → 3157
```

**Benefits**:
- More hierarchical feature learning
- Better abstraction

**Trade-offs**:
- Harder to train (vanishing gradients)
- More parameters (~2-3M)

### **Option 3: Increase Embedding Size** (LOW PRIORITY)
```python
# Current: embedding_size = 16
# Proposed: embedding_size = 32 or 64
```

**Benefits**:
- Better representation of discrete values (stack, effect_shown)
- More expressive embeddings

**Trade-offs**:
- Increases input dimension
- Might not address main bottleneck

### **Option 4: Action Space Reduction** (IF POSSIBLE)
- Reduce action space if many actions are redundant
- Use action masking more aggressively
- This addresses the root cause (large action space)

---

## Diagnostic Tests

### **Test 1: Check if Network is Saturated**
```python
# Monitor gradient magnitudes
# If gradients are very small, network might be saturated
# If gradients are large, network still has capacity to learn
```

### **Test 2: Compare Larger vs. Smaller Networks**
- Train with 480 → 512 → 256 → 3157
- Compare final win rates
- If larger network performs better, capacity was limiting

### **Test 3: Analyze Action Selection Patterns**
- Check if network struggles with similar actions
- See if it makes consistent mistakes
- If yes, action discrimination might be the issue

### **Test 4: Monitor Loss vs. Win Rate**
- If loss decreases but win rate plateaus → capacity limit
- If both plateau → other issues (exploration, rewards, etc.)

---

## Implementation Guide

### **Quick Test: Increase Hidden Width**
Add to `hyperparams_config.json`:
```json
{
  "network_architecture": {
    "hidden_layers": [512, 256],
    "comment": "Larger hidden layers for more capacity"
  }
}
```

Then modify `train_no_features.py` to use custom network:
```python
from torch import nn
custom_net = nn.Sequential(
    nn.Linear(input_length, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_actions)
)
model = NeuralNetwork(obs_space, embedding_size, num_actions, custom_net)
```

---

## Conclusion

**Current Assessment**: The network architecture is **likely a contributing factor** but probably not the **primary bottleneck**.

**Most Likely Issue**: The **128 → 3,157 compression** is very aggressive and could limit action discrimination.

**Recommended Action**: 
1. **First**: Fix other issues (target network sync, etc.) - already done ✅
2. **Second**: Try increasing hidden layer width (256 → 512, 128 → 256)
3. **Third**: If still plateauing, add another layer or increase embedding size

**Expected Improvement**: If capacity is the issue, increasing hidden width could improve win rate by 5-15% (depending on current plateau).

---

## References

- **Atari DQN**: ~1-2M parameters, 3-4 hidden layers
- **AlphaGo**: Much larger, but different problem
- **Card Game DQNs**: Typically 500K-2M parameters
- **Current**: 564K parameters - on the smaller side for this action space

**Verdict**: Network is **adequate but could benefit from more capacity**, especially in the hidden layers.

