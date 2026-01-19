---
title: Game-Based Dimension Rationale
tags: [neural-networks, architecture, game-mechanics, design-rationale]
created: 2026-01-16
related: [Game-Based Architecture, Dimension Selection, Network Architectures]
---

# Game-Based Dimension Rationale

## Overview

All network dimensions are chosen based on game structure characteristics, creating a direct mapping between architecture and game mechanics. This document explains the rationale for each dimension choice.

## Game Structure Fundamentals

### Card Structure
- **52 cards**: Standard deck (4 suits × 13 ranks)
- **13 ranks**: Ace (0) through King (12)
- **4 suits**: Clubs, Diamonds, Hearts, Spades

### Game Mechanics
- **Rank-based**: Scoring, scuttling, card effects depend on rank
- **Card-based**: All actions involve specific cards
- **Zone-based**: Cards exist in different zones (hand, field, deck, scrap, etc.)

## Dimension Choices

### Hidden Layer: 52 Neurons

**Rationale**: One neuron per card in the deck

**Game Alignment**:
- Direct correspondence: 52 cards → 52 neurons
- Each neuron can learn card-specific patterns
- Maintains card-level granularity

**Benefits**:
- Interpretable (each neuron = card)
- Sufficient capacity (52 distinct game units)
- Natural representation size

### Embedding Network Dimensions

#### embedding_dim = 52

**Rationale**: One dimension per card

**Game Alignment**:
- Each card gets its own embedding vector
- Maintains card-level representation
- Enables learning card-specific relationships

**Alternative Considered**: Smaller dimensions (e.g., 32)
- **Rejected**: Loses direct card correspondence
- **Chosen**: Maintains game structure alignment

#### zone_encoded_dim = 52

**Rationale**: One dimension per card after zone aggregation

**Game Alignment**:
- Maintains card-level granularity through preprocessing
- Each zone encoding preserves card information
- Fusion layer: 9 zones × 52 = 468 dimensions

**Alternative Considered**: Smaller (e.g., 32)
- **Rejected**: Would compress card information
- **Chosen**: Preserves card-level detail

### Multi-Encoder Network Dimensions

#### encoder_output_dim = 13

**Rationale**: One dimension per rank (Ace through King)

**Game Alignment**:
- Rank is fundamental to game mechanics:
  - Scoring: rank + 1 points
  - Scuttling: rank comparison
  - Card effects: rank-based
- Rank-level representation captures essential game structure
- More efficient than card-level (13 vs 52)

**Alternative Considered**: Card-level (52)
- **Rejected**: More parameters, rank is more fundamental
- **Chosen**: Rank-level is more aligned with game mechanics

#### encoder_hidden_dim = 26

**Rationale**: 2× expansion of rank dimension (2 × 13 = 26)

**Game Alignment**:
- Provides learning capacity (2× expansion is standard)
- Still game-based (multiple of 13)
- Balances capacity with structure alignment

**Alternative Considered**: 
- **13**: Too small, insufficient capacity
- **52**: Too large, loses rank-level focus
- **Chosen**: 26 provides capacity while maintaining rank-based structure

**Fusion Layer**: 9 zones × 13 = 117 dimensions (rank-level representation)

## Comparison: Card-Level vs Rank-Level

### Embedding Network (Card-Level)

**Representation**: Card-level (52 dimensions)
- **Advantages**:
  - Preserves all card information
  - Can learn card-specific patterns
  - More granular representation
- **Disadvantages**:
  - More parameters
  - May learn redundant information (4 cards per rank)

### Multi-Encoder Network (Rank-Level)

**Representation**: Rank-level (13 dimensions)
- **Advantages**:
  - Aligns with game mechanics (rank is fundamental)
  - Fewer parameters
  - More efficient representation
- **Disadvantages**:
  - Loses suit information
  - Less granular than card-level

**Design Choice**: Different networks use different granularities to test both approaches.

## Efficiency Hypothesis

### Parameter Comparison

| Network | Dimensions | Parameters | Ratio vs Boolean |
|---------|-----------|------------|------------------|
| Boolean | 468→512→256 | ~1.18M | 1.0× |
| Embedding | 52×52 (card-level) | ~197K | 0.17× (6× smaller) |
| Multi-Encoder | 26×13 (rank-level) | ~189K | 0.16× (6× smaller) |

### Research Question

**Can game-aligned networks with 5-6× fewer parameters achieve comparable or better performance?**

This tests the **efficiency hypothesis**: Structure-aware design enables efficiency.

## Why Game-Based Dimensions?

### 1. Interpretability

Dimensions correspond to game units:
- 52 neurons = 52 cards
- 13 dimensions = 13 ranks
- Easy to understand and analyze

### 2. Structure Alignment

Architecture reflects game mechanics:
- Card-level: Embedding network
- Rank-level: Multi-encoder network
- Both align with how the game works

### 3. Efficiency Testing

Smaller networks test efficiency:
- If they perform well: Strong result
- If they don't: Capacity vs preprocessing analysis
- Either way: Provides insights

### 4. Thesis Contribution

Demonstrates:
- Structure-aware design value
- Efficiency through alignment
- Importance of domain knowledge

## Alternative Approaches (For Comparison)

See [[Dimension Strategy Comparison]] for:
- Parameter matching (fair comparison)
- Compression ratios (information control)
- Information-theoretic (principled balance)

All alternatives are documented for future experiments and thesis comparison.

## Implementation

### Default Dimensions

All networks use game-based dimensions by default:

```python
# Embedding network
EmbeddingBasedNetwork(
    observation_space,
    embedding_dim=52,        # Game-based: one per card
    zone_encoded_dim=52      # Game-based: one per card
)

# Multi-encoder network
MultiEncoderNetwork(
    observation_space,
    encoder_hidden_dim=26,  # Game-based: 2×13 ranks
    encoder_output_dim=13    # Game-based: one per rank
)
```

### Changing Dimensions

Use `cuttle.network_dimensions` module to calculate alternative dimensions:

```python
from cuttle.network_dimensions import recommend_dimensions

# Get parameter-matched dimensions
dims = recommend_dimensions(
    method="embedding",
    strategy="parameter_matching"
)
```

## Related Concepts

- [[Game-Based Architecture]] - Overall architecture design
- [[Dimension Selection]] - All dimension strategies
- [[Dimension Strategy Comparison]] - Comprehensive comparison
- [[Network Architectures]] - Network structure details

---
*Rationale for game-based dimension choices*
