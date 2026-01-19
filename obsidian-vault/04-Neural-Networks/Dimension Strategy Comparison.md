---
title: Dimension Strategy Comparison
tags: [neural-networks, experiments, research-design, thesis]
created: 2026-01-16
related: [Dimension Selection, Network Architectures, Input Representation Experiments]
---

# Dimension Strategy Comparison

## Overview

This document provides a comprehensive comparison of dimension selection strategies for future thesis analysis and potential follow-up experiments. The current implementation uses **game-based dimensions**, but all strategies are documented here for comparison.

## Strategy Summary

| Strategy | Goal | Key Principle | Use Case |
|----------|------|---------------|----------|
| **Game-Based** (Current) | Align with game structure | Dimensions = game units (52 cards, 13 ranks) | Efficiency hypothesis, interpretability |
| **Parameter Matching** | Fair capacity comparison | Match parameter counts across networks | Isolate preprocessing effects |
| **Compression Ratio** | Control information flow | Explicit compression ratio | Information preservation studies |
| **Information-Theoretic** | Balance info/complexity | Effective dimensionality | Principled balance |

## Detailed Comparison

### Embedding-Based Network

#### Game-Based (Current Choice)
- **Dimensions**: `embedding_dim=52, zone_encoded_dim=52`
- **Parameters**: ~197K
- **Rationale**: One dimension per card (52 cards in deck)
- **Advantages**:
  - Direct game structure alignment
  - Interpretable (each dimension = card)
  - Tests efficiency (5-6× fewer params)
- **Disadvantages**:
  - May be capacity-limited
  - Cannot directly compare with baseline (different param counts)

#### Parameter Matching
- **Dimensions**: `embedding_dim=128, zone_encoded_dim=128`
- **Parameters**: ~250K (closer to boolean baseline)
- **Rationale**: Match boolean network capacity
- **Advantages**:
  - Fair comparison (same capacity)
  - Isolates preprocessing as variable
  - Stronger experimental control
- **Disadvantages**:
  - Less efficient (more parameters)
  - Doesn't test efficiency hypothesis
- **When to use**: Follow-up experiment to isolate preprocessing effects

#### Compression Ratio (1.0)
- **Dimensions**: `embedding_dim=52, zone_encoded_dim=52`
- **Parameters**: ~197K
- **Rationale**: No compression (fusion_dim = input_dim)
- **Advantages**:
  - Preserves all information
  - Matches input dimensionality
- **Disadvantages**:
  - May be redundant with game-based
- **When to use**: Information preservation studies

#### Information-Theoretic (0.5)
- **Dimensions**: `embedding_dim=36, zone_encoded_dim=26`
- **Parameters**: ~182K
- **Rationale**: Preserve 50% of information
- **Advantages**:
  - Principled compression
  - Balanced approach
- **Disadvantages**:
  - Less interpretable
  - Arbitrary information ratio
- **When to use**: When exploring information-content trade-offs

### Multi-Encoder Network

#### Game-Based (Current Choice)
- **Dimensions**: `encoder_hidden_dim=26, encoder_output_dim=13`
- **Parameters**: ~189K
- **Rationale**: Rank-level representation (13 ranks), 2× expansion for capacity
- **Advantages**:
  - Rank-level interpretability
  - Aligns with game mechanics (rank is fundamental)
  - Tests efficiency (5-6× fewer params)
- **Disadvantages**:
  - May be capacity-limited
  - Rank-level may lose card-specific information
- **Note**: Different from embedding (rank vs card level)

#### Parameter Matching
- **Dimensions**: `encoder_hidden_dim=128, encoder_output_dim=64`
- **Parameters**: ~333K
- **Rationale**: Match boolean network capacity
- **Advantages**:
  - Fair comparison
  - Isolates preprocessing
- **Disadvantages**:
  - More parameters
  - Doesn't test efficiency
- **When to use**: Follow-up experiment

#### Compression Ratio (1.0)
- **Dimensions**: `encoder_hidden_dim=104, encoder_output_dim=52`
- **Parameters**: ~290K
- **Rationale**: No compression
- **Advantages**:
  - Preserves information
  - Card-level representation
- **Disadvantages**:
  - More parameters than game-based
- **When to use**: Information preservation with card-level encoding

#### Information-Theoretic (0.5)
- **Dimensions**: `encoder_hidden_dim=52, encoder_output_dim=26`
- **Parameters**: ~217K
- **Rationale**: 50% information preservation
- **Advantages**:
  - Balanced approach
  - Moderate compression
- **Disadvantages**:
  - Less interpretable
- **When to use**: Information-content studies

## Current Experimental Design

### Primary Experiment: Efficiency Hypothesis

**Question**: Can game-based networks (5-6× fewer parameters) achieve comparable or better performance?

**Networks**:
- Boolean: ~1.18M parameters (baseline)
- Embedding: ~197K parameters (game-based: 52×52)
- Multi-Encoder: ~189K parameters (game-based: 26×13)

**Hypothesis**: Smaller game-aligned networks will perform comparably or better, demonstrating efficiency through structure-aware design.

**Why This Is Interesting**:
- If true: Strong result showing efficiency through design
- If false: Can analyze whether it's capacity or preprocessing
- Either way: Provides insights for follow-up experiments

### Potential Follow-Up Experiments

#### Experiment 2: Preprocessing Isolation

**Question**: Which preprocessing strategy is best when capacity is equal?

**Design**: Use parameter matching to make all networks similar size, then compare preprocessing strategies.

**Networks**:
- Boolean: ~1.18M (baseline)
- Embedding: ~250K (parameter-matched: 128×128)
- Multi-Encoder: ~333K (parameter-matched: 128×64)

**Hypothesis**: With equal capacity, preprocessing differences will be more apparent.

#### Experiment 3: Information Preservation

**Question**: How does information compression affect performance?

**Design**: Vary compression ratios and information ratios.

**Networks**: Multiple variants with different compression/information ratios.

**Hypothesis**: Moderate compression (0.5-1.0) preserves performance while reducing parameters.

## Parameter Count Analysis

### Current Setup (Game-Based)

| Network | Parameters | Ratio vs Boolean |
|---------|------------|------------------|
| Boolean | 1,182,805 | 1.0× (baseline) |
| Embedding | 197,169 | 0.17× (6× smaller) |
| Multi-Encoder | 189,018 | 0.16× (6× smaller) |

### Parameter-Matched Alternative

| Network | Parameters | Ratio vs Boolean |
|---------|------------|------------------|
| Boolean | 1,182,805 | 1.0× (baseline) |
| Embedding | 250,445 | 0.21× (5× smaller) |
| Multi-Encoder | 332,685 | 0.28× (3.5× smaller) |

## Research Implications

### If Game-Based Networks Perform Well

**Strong Result**: Demonstrates that:
1. Structure-aware design enables efficiency
2. Preprocessing matters more than raw capacity
3. Game alignment improves learning

**Thesis Contribution**: "Our game-based architectures achieve comparable performance with 5-6× fewer parameters, demonstrating the importance of structure-aware design over brute-force scaling."

### If Game-Based Networks Perform Poorly

**Analysis Needed**:
1. Is it capacity limitation or preprocessing?
2. Would parameter-matched versions perform better?
3. What's the minimum capacity needed?

**Thesis Contribution**: "While game-based alignment shows promise, additional capacity is needed. Parameter-matched comparisons reveal preprocessing effects are capacity-dependent."

### If Results Are Mixed

**Rich Analysis Opportunity**:
1. Which preprocessing works best at different capacities?
2. What's the efficiency frontier?
3. How do game-based dimensions compare to arbitrary dimensions?

**Thesis Contribution**: "Our analysis reveals a complex interaction between preprocessing strategy and network capacity, with game-based dimensions providing efficiency advantages at lower parameter counts."

## Implementation for Future Experiments

### Switching Strategies

All strategies are implemented in `cuttle.network_dimensions`:

```python
from cuttle.network_dimensions import recommend_dimensions

# Game-based (current)
dims = recommend_dimensions(method="embedding", strategy="game_based")

# Parameter matching (for follow-up)
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

### Analysis Script

Use `scripts/analyze_dimensions.py` to compare all strategies:

```bash
python scripts/analyze_dimensions.py
```

## Recommendations for Thesis

### Primary Experiment (Current)

**Use game-based dimensions** to test efficiency hypothesis:
- Stronger result if successful
- Tests practical question: "Can we do better with less?"
- Demonstrates structure-aware design value

### Future Experiments

1. **Parameter-matched comparison**: If game-based performs poorly, test whether it's capacity
2. **Compression studies**: Explore information preservation trade-offs
3. **Ablation studies**: Compare game-based vs arbitrary dimensions

### Documentation

Document all strategies in thesis:
- Explain why game-based was chosen
- Present parameter-matched alternatives
- Discuss trade-offs and future directions

## Related Concepts

- [[Dimension Selection]] - Detailed strategy descriptions
- [[Network Architectures]] - Network structure details
- [[Game-Based Architecture]] - Game-based design rationale
- [[Input Representation Experiments]] - Experimental design

---
*Comprehensive comparison of dimension selection strategies for thesis analysis*
