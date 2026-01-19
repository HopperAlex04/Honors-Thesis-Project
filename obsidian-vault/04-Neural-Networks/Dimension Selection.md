---
title: Dimension Selection
tags: [neural-networks, architecture, hyperparameters, design]
created: 2026-01-16
related: [Network Architectures, Game-Based Architecture, Hyperparameters]
---

# Dimension Selection

## Overview

Choosing encoder dimensions is a critical design decision that affects network capacity, learning efficiency, and parameter count. This document describes rigorous methods for selecting dimensions in the embedding-based and multi-encoder networks.

## Current Dimensions (Game-Based)

**All networks use game-based dimensions by default**, aligning architecture with game structure.

### Embedding-Based Network
- **embedding_dim**: 52 (one per card in deck)
- **zone_encoded_dim**: 52 (one per card, card-level representation)
- **Total parameters**: ~197K
- **Rationale**: Each dimension corresponds to a card, maintaining direct game structure alignment

### Multi-Encoder Network
- **encoder_hidden_dim**: 26 (2×13 ranks, provides capacity)
- **encoder_output_dim**: 13 (one per rank: Ace through King)
- **Total parameters**: ~189K
- **Rationale**: Rank-level representation (13 ranks) with 2× expansion for learning capacity

### Boolean Network (Baseline)
- **Architecture**: 468 → 512 → 256 → num_actions
- **Total parameters**: ~1.18M
- **Note**: Still uses original architecture (not updated to 52-neuron hidden layer)

## Dimension Selection Strategies

### 1. Parameter Count Matching

**Goal**: Match parameter count across network types for fair comparison.

**Method**: Solve for dimensions that yield similar total parameter counts.

**Formula** (Embedding Network):
```
Parameters = 52×emb + (emb+1)×zone + (9×zone+1)×52 + (52+1)×actions
```

**Formula** (Multi-Encoder Network):
```
Parameters = 9×[(52+1)×hidden + (hidden+1)×output] + (9×output+1)×52 + (52+1)×actions
```

**Use Case**: When you want to control for parameter count as a variable.

**Example**:
```python
from cuttle.network_dimensions import recommend_dimensions

# Match boolean network parameter count
dims = recommend_dimensions(
    method="embedding",
    strategy="parameter_matching",
    target_params=1182805  # Boolean network params
)
```

### 2. Compression Ratio

**Goal**: Control information compression from input to fusion layer.

**Method**: Set fusion dimension based on desired compression ratio.

**Formula**:
```
compression_ratio = input_dim / fusion_dim
fusion_dim = input_dim / compression_ratio
```

**Compression Ratios**:
- **1.0**: No compression (fusion_dim = input_dim)
- **< 1.0**: Expansion (fusion_dim > input_dim)
- **> 1.0**: Compression (fusion_dim < input_dim)

**Use Case**: When you want to control information preservation.

**Example**:
```python
# No compression
dims = recommend_dimensions(
    method="embedding",
    strategy="compression_ratio",
    compression_ratio=1.0
)

# 2x compression
dims = recommend_dimensions(
    method="embedding",
    strategy="compression_ratio",
    compression_ratio=2.0
)
```

### 3. Game-Based Heuristics

**Goal**: Align dimensions with game structure.

**Method**: Use game structure to inform dimensions.

**Embedding Network**:
- **embedding_dim = 52**: One per card
- **zone_encoded_dim = 52**: One per card (card-level representation)

**Multi-Encoder Network**:
- **encoder_output_dim = 13**: One per rank (rank-level representation)
- **encoder_hidden_dim = 26**: 2× output for capacity

**Rationale**: Dimensions correspond to game units (cards, ranks).

**Use Case**: When you want architecture to reflect game structure.

**Example**:
```python
dims = recommend_dimensions(
    method="embedding",
    strategy="game_based"
)
# Returns: {"embedding_dim": 52, "zone_encoded_dim": 52}
```

### 4. Information-Theoretic

**Goal**: Preserve a certain ratio of information from input.

**Method**: Calculate effective dimensionality considering sparsity and information content.

**Formula**:
```
effective_dim = input_dim × information_ratio
zone_encoded_dim = effective_dim / num_zones
```

**Information Ratios**:
- **0.5**: Preserve 50% of information (moderate compression)
- **0.3**: Preserve 30% of information (aggressive compression)
- **0.7**: Preserve 70% of information (conservative)

**Use Case**: When you want to balance information preservation with model size.

**Example**:
```python
dims = recommend_dimensions(
    method="embedding",
    strategy="information_theoretic",
    information_ratio=0.5
)
```

## Comparison of Strategies

### Embedding Network

| Strategy | embedding_dim | zone_encoded_dim | Parameters | Status |
|----------|---------------|-------------------|------------|--------|
| **Game-Based** | **52** | **52** | **~197K** | **Current (Default)** |
| Parameter Matching | 128 | 128 | ~250K | Alternative |
| Compression (1.0) | 52 | 52 | ~197K | Alternative (same as game-based) |
| Information (0.5) | 36 | 26 | ~182K | Alternative |

### Multi-Encoder Network

| Strategy | encoder_hidden_dim | encoder_output_dim | Parameters | Status |
|----------|-------------------|-------------------|------------|--------|
| **Game-Based** | **26** | **13** | **~189K** | **Current (Default)** |
| Parameter Matching | 128 | 64 | ~333K | Alternative |
| Compression (1.0) | 104 | 52 | ~290K | Alternative |
| Information (0.5) | 52 | 26 | ~217K | Alternative |

**Note**: Game-based uses rank-level representation (13 ranks) rather than card-level (52 cards), reflecting that rank is fundamental to game mechanics.

## Design Decision: Game-Based Dimensions (Current Choice)

**We use game-based dimensions as the default** for the following reasons:

### Rationale

1. **Interpretability**: Dimensions directly correspond to game units (cards, ranks)
2. **Structure Alignment**: Architecture reflects game mechanics
3. **Efficiency Focus**: Smaller networks (5-6× fewer parameters) test efficiency hypothesis
4. **Thesis Contribution**: If smaller game-based networks perform well, demonstrates efficiency through structure-aware design

### Game Structure Mapping

- **52 cards**: Embedding dimension, zone encoding dimension, hidden layer size
- **13 ranks**: Multi-encoder output dimension (rank-level representation)
- **4 suits**: Could be used for future variations (not currently used)

### Efficiency Hypothesis

By using game-based dimensions, we test:
> "Can networks with 5-6× fewer parameters (but game-aligned structure) achieve comparable or better performance than larger baseline networks?"

This is a stronger research contribution than simply matching parameter counts.

## Alternative Strategies (For Future Comparison)

All strategies are documented here for future experiments and comparison:

### For Fair Parameter Comparison

**Use Parameter Matching**: Ensures all networks have similar capacity, isolating preprocessing as the only variable.

**When to use**: If you want to test pure preprocessing effects without capacity differences.

**Recommended dimensions**:
- Embedding: `embedding_dim=128, zone_encoded_dim=128` (~250K params)
- Multi-Encoder: `encoder_hidden_dim=128, encoder_output_dim=64` (~333K params)

### For Information Preservation

**Use Compression Ratio**: Explicitly controls how much information flows from input to hidden layer.

**When to use**: When you want to control information preservation explicitly.

**Recommended dimensions** (compression_ratio=1.0):
- Embedding: `embedding_dim=52, zone_encoded_dim=52` (~197K params)
- Multi-Encoder: `encoder_hidden_dim=104, encoder_output_dim=52` (~290K params)

### For Balanced Approach

**Use Information-Theoretic**: Balances information preservation with model complexity.

**When to use**: When you want principled balance between information and complexity.

**Recommended dimensions** (information_ratio=0.5):
- Embedding: `embedding_dim=36, zone_encoded_dim=26` (~182K params)
- Multi-Encoder: `encoder_hidden_dim=52, encoder_output_dim=26` (~217K params)

## Implementation

### Using the Utility Module

```python
from cuttle.network_dimensions import recommend_dimensions, print_dimension_analysis

# Get recommendations
dims = recommend_dimensions(
    method="embedding",
    strategy="parameter_matching",
    target_params=1182805
)

# Print analysis
print_dimension_analysis("embedding")
print_dimension_analysis("multi_encoder")
```

### Integration with Networks

Dimensions can be calculated and passed to network constructors:

```python
from cuttle.network_dimensions import recommend_dimensions
from cuttle.networks import EmbeddingBasedNetwork

# Calculate dimensions
dims = recommend_dimensions(
    method="embedding",
    strategy="game_based"
)

# Create network
model = EmbeddingBasedNetwork(
    observation_space,
    embedding_dim=dims["embedding_dim"],
    zone_encoded_dim=dims["zone_encoded_dim"],
    num_actions=num_actions
)
```

## Trade-offs

### Larger Dimensions

**Advantages**:
- More representational capacity
- Can learn more complex patterns
- Better for complex strategies

**Disadvantages**:
- More parameters (risk of overfitting)
- Slower training
- More memory usage

### Smaller Dimensions

**Advantages**:
- Fewer parameters (less overfitting risk)
- Faster training
- Less memory usage

**Disadvantages**:
- Less representational capacity
- May struggle with complex patterns
- Potential bottleneck

## Related Concepts

- [[Network Architectures]] - Network structure details
- [[Game-Based Architecture]] - Game-based design rationale
- [[Hyperparameters]] - Configuration and tuning
- [[Neural Network Basics]] - Fundamental concepts

---
*Rigorous dimension selection for network architectures*
