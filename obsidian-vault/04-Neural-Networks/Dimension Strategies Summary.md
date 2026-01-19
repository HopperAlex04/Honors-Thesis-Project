---
title: Dimension Strategies Summary
tags: [neural-networks, experiments, research, thesis]
created: 2026-01-16
related: [Dimension Selection, Dimension Strategy Comparison, Game-Based Dimension Rationale]
---

# Dimension Strategies Summary

## Quick Reference

This document provides a quick reference for all dimension selection strategies, their rationale, and when to use them for thesis experiments and analysis.

## Current Choice: Game-Based Dimensions

**All networks use game-based dimensions by default.**

### Embedding Network
- **embedding_dim**: 52 (one per card)
- **zone_encoded_dim**: 52 (one per card)
- **Parameters**: ~197K
- **Rationale**: Card-level representation aligned with game structure

### Multi-Encoder Network
- **encoder_hidden_dim**: 26 (2×13 ranks)
- **encoder_output_dim**: 13 (one per rank)
- **Parameters**: ~189K
- **Rationale**: Rank-level representation (rank is fundamental to game mechanics)

### Efficiency Hypothesis
Both networks have **5-6× fewer parameters** than boolean baseline, testing whether structure-aware design enables efficiency.

## All Strategies

### 1. Game-Based (Current)

**Goal**: Align dimensions with game structure

**Dimensions**:
- Embedding: 52×52 (card-level)
- Multi-Encoder: 26×13 (rank-level)

**Use When**:
- Testing efficiency hypothesis
- Want interpretable architecture
- Structure alignment is priority

**Thesis Value**: Strong result if smaller networks perform well

### 2. Parameter Matching

**Goal**: Match parameter counts for fair comparison

**Dimensions**:
- Embedding: 128×128 (~250K params)
- Multi-Encoder: 128×64 (~333K params)

**Use When**:
- Want to isolate preprocessing effects
- Need fair capacity comparison
- Follow-up experiment

**Thesis Value**: Isolates preprocessing as variable

### 3. Compression Ratio

**Goal**: Control information compression

**Dimensions** (compression_ratio=1.0):
- Embedding: 52×52 (~197K params)
- Multi-Encoder: 104×52 (~290K params)

**Use When**:
- Studying information preservation
- Need explicit compression control
- Information-theoretic analysis

**Thesis Value**: Tests information preservation trade-offs

### 4. Information-Theoretic

**Goal**: Balance information and complexity

**Dimensions** (information_ratio=0.5):
- Embedding: 36×26 (~182K params)
- Multi-Encoder: 52×26 (~217K params)

**Use When**:
- Principled balance needed
- Exploring information-content trade-offs
- Theoretical analysis

**Thesis Value**: Principled dimension selection

## Comparison Table

| Strategy | Embedding Dims | Multi-Encoder Dims | Embed Params | Multi Params | Use Case |
|----------|---------------|-------------------|-------------|--------------|----------|
| **Game-Based** | 52×52 | 26×13 | 197K | 189K | **Current: Efficiency** |
| Parameter Match | 128×128 | 128×64 | 250K | 333K | Fair comparison |
| Compression (1.0) | 52×52 | 104×52 | 197K | 290K | Info preservation |
| Information (0.5) | 36×26 | 52×26 | 182K | 217K | Balanced approach |

## Research Questions by Strategy

### Game-Based (Current Experiment)

**Question**: Can structure-aware networks with 5-6× fewer parameters achieve comparable/better performance?

**Hypothesis**: Yes - game alignment enables efficiency

**If True**: Strong result demonstrating efficiency through design

**If False**: Analyze capacity vs preprocessing

### Parameter Matching (Follow-Up)

**Question**: Which preprocessing strategy is best when capacity is equal?

**Hypothesis**: Preprocessing differences will be more apparent

**Use**: If game-based performs poorly, test whether it's capacity

### Compression Ratio (Information Study)

**Question**: How does information compression affect performance?

**Hypothesis**: Moderate compression preserves performance

**Use**: Information-theoretic analysis

### Information-Theoretic (Balance Study)

**Question**: What's the optimal information/complexity trade-off?

**Hypothesis**: Moderate ratios (0.5) provide good balance

**Use**: Theoretical dimension analysis

## Implementation

### Current (Game-Based)

```python
# Uses defaults (game-based)
model = EmbeddingBasedNetwork(observation_space, num_actions=actions)
model = MultiEncoderNetwork(observation_space, num_actions=actions)
```

### Alternative Strategies

```python
from cuttle.network_dimensions import recommend_dimensions

# Parameter matching
dims = recommend_dimensions(
    method="embedding",
    strategy="parameter_matching",
    target_params=1182805
)

# Compression ratio
dims = recommend_dimensions(
    method="embedding",
    strategy="compression_ratio",
    compression_ratio=1.0
)

# Information-theoretic
dims = recommend_dimensions(
    method="embedding",
    strategy="information_theoretic",
    information_ratio=0.5
)
```

## Thesis Documentation

### Primary Experiment

**Strategy**: Game-Based
**Rationale**: Tests efficiency hypothesis, structure alignment
**Expected Contribution**: Efficiency through structure-aware design

### Future Experiments

Document all strategies for:
- Follow-up experiments
- Ablation studies
- Comparative analysis
- Thesis discussion

### Analysis Plan

1. **Primary**: Compare game-based networks (current)
2. **Follow-up**: Parameter-matched comparison (if needed)
3. **Analysis**: Discuss all strategies and trade-offs
4. **Conclusion**: Which approach works best and why

## Related Documents

- [[Dimension Selection]] - Detailed strategy descriptions
- [[Dimension Strategy Comparison]] - Comprehensive comparison
- [[Game-Based Dimension Rationale]] - Rationale for current choice
- [[Game-Based Architecture]] - Overall architecture design

---
*Quick reference for dimension selection strategies*
