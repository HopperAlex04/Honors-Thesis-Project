---
title: Game-Based Architecture
tags: [neural-networks, architecture, design, game-mechanics]
created: 2026-01-16
related: [Network Architectures, Neural Network Basics, CuttleEnvironment]
---

# Game-Based Architecture

## Overview

The neural network architectures in this project use a **game-based hidden layer design** where the hidden layer size directly corresponds to game structure: **52 neurons, one per card in a standard deck**.

## Design Rationale

### Why 52 Neurons for Hidden Layer?

**Direct Correspondence to Game Structure**:
- Standard deck: 52 cards
- Hidden layer: 52 neurons
- Each neuron can learn patterns related to a specific card

**Game Mechanics Alignment**:
- Cards are fundamental game units
- All game actions involve cards
- Card relationships drive strategy
- 52 neurons provide natural representation capacity

### Why Game-Based Dimensions for Preprocessing?

**Embedding Network**:
- **embedding_dim = 52**: One dimension per card (card-level representation)
- **zone_encoded_dim = 52**: One dimension per card after zone aggregation
- **Rationale**: Maintains card-level granularity throughout preprocessing

**Multi-Encoder Network**:
- **encoder_output_dim = 13**: One dimension per rank (Ace through King)
- **encoder_hidden_dim = 26**: 2× expansion (2×13 ranks) for learning capacity
- **Rationale**: Rank is fundamental to game mechanics (scoring, scuttling, card effects)
- **Note**: Different from embedding - uses rank-level rather than card-level representation

### Benefits

1. **Interpretability**: Each neuron corresponds to a card
2. **Game-Aligned**: Architecture reflects game structure
3. **Sufficient Capacity**: 52 neurons provide reasonable representational power
4. **Simple Design**: Single hidden layer keeps architecture simple

## Architecture Structure

### Shared Hidden Layer

**All networks share**:
```
Preprocessed Input (variable dimension)
    ↓
52-Neuron Hidden Layer (game-based)
    ↓
num_actions Output (Q-values)
```

### Hidden Layer Details

- **Size**: 52 neurons (one per card)
- **Activation**: ReLU
- **Input**: Variable dimension (depends on preprocessing)
- **Output**: 52-dimensional representation

### Output Layer

- **Size**: 52 → num_actions
- **Activation**: None (unbounded Q-values)
- **Purpose**: Map 52-neuron representation to action Q-values

## Experimental Design

### Constant Architecture

By keeping the hidden layer constant (52 neurons) across all network types:
- **Fair Comparison**: Performance differences due to preprocessing, not architecture
- **Isolated Variable**: Preprocessing complexity is the experimental variable
- **Controlled Experiment**: Architecture differences don't confound results

### Preprocessing Variation

Different preprocessing strategies feed into the same 52-neuron layer:

1. **Boolean**: 468-dim → 52 neurons
2. **Embedding**: ~468-dim → 52 neurons
3. **Multi-Encoder**: ~288-dim → 52 neurons

All converge to the same hidden representation size.

## Design Philosophy

### Complexity Distribution

**Preprocessing**: Where complexity and variation live
- Boolean: Simple concatenation
- Embedding: Learned representations
- Multi-Encoder: Zone-specific learning

**Hidden Layer**: Simple and constant
- 52 neurons (game-based)
- Same across all networks
- Enables fair comparison

### Why Not Larger?

**52 neurons is sufficient** because:
- Game has 52 distinct cards
- Each neuron can represent card-related patterns
- Additional neurons may not add value
- Simpler architecture is easier to analyze

### Why Not Smaller?

**52 neurons provides capacity** for:
- Learning card relationships
- Representing game state patterns
- Mapping to action space
- Sufficient expressiveness without overfitting

## Implementation

### Code Structure

```python
# Shared hidden layer (all networks)
self.hidden_layer = nn.Sequential(
    nn.Linear(preprocessed_dim, 52),  # Variable input → 52 neurons (game-based)
    nn.ReLU()
)

# Shared output layer (all networks)
self.output_layer = nn.Linear(52, num_actions)  # 52 → num_actions
```

### Game-Based Preprocessing Dimensions

**Embedding Network**:
```python
# Card embeddings: 52 cards → 52 dimensions (one per card)
self.card_embedding = nn.Embedding(52, embedding_dim=52)

# Zone aggregator: 52 → 52 (maintains card-level representation)
self.zone_aggregator = nn.Sequential(
    nn.Linear(52, 52),  # zone_encoded_dim = 52
    nn.ReLU()
)
```

**Multi-Encoder Network**:
```python
# Zone encoders: 52 (cards) → 26 (2×ranks) → 13 (ranks)
self.hand_encoder = nn.Sequential(
    nn.Linear(52, 26),      # encoder_hidden_dim = 26 (2×13)
    nn.ReLU(),
    nn.Linear(26, 13)       # encoder_output_dim = 13 (one per rank)
)
```

### Network-Specific Preprocessing

Each network type implements different preprocessing:
- **Boolean**: Concatenation → 468-dim
- **Embedding**: Embeddings + aggregation → ~468-dim
- **Multi-Encoder**: Zone encoders → ~288-dim

All feed into the same 52-neuron hidden layer.

## Analysis

### What Do the 52 Neurons Learn?

Each neuron can learn:
- Card-specific patterns
- Card relationships
- Strategic importance of cards
- Context-dependent card values

### Interpretation

- **Neuron 0**: May learn patterns related to Ace of Clubs
- **Neuron 1**: May learn patterns related to Ace of Diamonds
- **Neuron 51**: May learn patterns related to King of Spades

(Note: Actual learned patterns depend on training and may not directly map to specific cards)

## Related Concepts

- [[Network Architectures]] - Three network types using this architecture
- [[Input Representation Experiments]] - Experimental design
- [[Neural Network Basics]] - Fundamental concepts
- [[CuttleEnvironment]] - Game structure that inspired this design

---
*Game-based architecture design for Cuttle DQN project*
