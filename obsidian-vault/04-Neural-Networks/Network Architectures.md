---
title: Network Architectures
tags: [neural-networks, architecture, dqn, input-representation]
created: 2026-01-16
related: [Neural Network Basics, Deep Q-Network, Observation Space, Reward Engineering]
---

# Network Architectures

## Overview

This project implements three neural network architectures for DQN agents, all sharing a common game-based hidden layer design. The architectures differ in their **preprocessing** of observations, allowing comparison of input representation strategies.

## Shared Architecture: Game-Based Hidden Layer

**All networks share a simple, game-based hidden layer:**

- **Hidden Layer Size**: 52 neurons (one per card in the deck)
- **Rationale**: Directly corresponds to the 52 cards in a standard deck
- **Structure**: `preprocessed_input → 52 neurons → num_actions`
- **Activation**: ReLU on hidden layer, no activation on output (unbounded Q-values)

### Design Philosophy

- **Complexity in preprocessing**: All complexity and variation is in the preprocessing/input representation
- **Simple hidden layer**: Shared 52-neuron layer keeps architecture constant across all input types
- **Fair comparison**: Differences in performance will be due to input representation, not architecture
- **Preprocessing flexibility**: Preprocessing can output different dimensions (not required to match)

### Architecture Flow

```
Observation Dict (same for all)
    ↓
Preprocessing (varies by network type)
    ↓
Preprocessed Tensor (variable dimension)
    ↓
52-Neuron Hidden Layer (shared, game-based)
    ↓
num_actions Output (Q-values)
```

## Network Types

### 1. Boolean Network (NeuralNetwork)

**Strategy**: Simple concatenation of boolean arrays

**Architecture**:
```
468-dim concatenation → 52 neurons → num_actions
```

**Preprocessing**:
- Concatenates all 9 boolean zone arrays (52-length each)
- Total: 468 dimensions
- No learned representations

**Use Case**: Baseline for comparison

**Configuration**: `network_type: "boolean"`

### 2. Embedding-Based Network (EmbeddingBasedNetwork)

**Strategy**: Learned card embeddings with zone aggregation

**Architecture**:
```
Card Embeddings → Zone Aggregation → Fusion → 52 neurons → num_actions
```

**Preprocessing**:
1. **Card Embedding Layer**: Embed each of 52 cards into `embedding_dim` vectors (default: 52)
2. **Zone Embedding**: For each zone, create embeddings for present cards
3. **Zone Aggregation**: Max pooling to aggregate cards in each zone
4. **Zone Combination**: Concatenate aggregated zone representations
5. **Output Dimension**: `9 zones × zone_encoded_dim` (default: 9 × 52 = 468)

**Key Components**:
- `nn.Embedding(52, embedding_dim)` for card embeddings
- Max pooling per zone (aggregates variable number of cards)
- Zone aggregator: `embedding_dim → zone_encoded_dim`
- Fusion: Concatenate all zone encodings

**Advantages**:
- Learns card relationships and similarities
- Can capture card synergies
- More parameter-efficient for sparse zones
- Attention-like behavior through max pooling

**Configuration**: `network_type: "embedding"`, `embedding_dim: 52`

### 3. Multi-Encoder Network (MultiEncoderNetwork)

**Strategy**: Separate encoders for each zone type

**Architecture**:
```
Zone Encoders → Fusion → 52 neurons → num_actions
```

**Preprocessing**:
1. **Zone-Specific Encoders**: Separate small networks for each zone type:
   - Hand encoder
   - Field encoder
   - Revealed encoder
   - Off-field encoder
   - Off-revealed encoder
   - Deck encoder
   - Scrap encoder
   - Stack encoder
   - Effect-shown encoder

2. **Zone Encoding**: Each encoder processes its boolean array (52-dim) → encoded representation
3. **Fusion Layer**: Concatenate all zone encodings
4. **Output Dimension**: `9 zones × encoder_output_dim` (default: 9 × 32 = 288)

**Key Components**:
- 9 separate encoders (one per zone type)
- Each encoder: `52 → 64 → 32` (default)
- Fusion: Concatenation of all zone encodings

**Advantages**:
- Zone-specific feature learning
- Can learn different patterns per zone type
- Modular architecture
- Potentially better credit assignment

**Configuration**: `network_type: "multi_encoder"`

## Comparison

| Network Type | Preprocessing Complexity | Input Dimension | Parameters (current) |
|-------------|-------------------------|-----------------|---------------------|
| Boolean | Low (concatenation) | 468 | ~1.18M |
| Embedding | Medium (embeddings + pooling) | 468 | ~197K |
| Multi-Encoder | High (9 separate encoders) | 288 | ~232K |

**Note**: Dimensions can be adjusted using rigorous methods. See [[Dimension Selection]] for strategies to choose optimal dimensions based on parameter matching, compression ratios, game-based heuristics, or information-theoretic principles.

## Experimental Design

### Variable: Preprocessing Complexity

The experimental variable is **preprocessing/input representation complexity**:
- Boolean: Simple concatenation (baseline)
- Embedding: Learned card embeddings
- Multi-Encoder: Zone-specific encoders

### Constant: Hidden Layer Architecture

All networks share the same hidden layer:
- 52 neurons (game-based)
- ReLU activation
- Same output layer (52 → num_actions)

This ensures fair comparison - performance differences are due to input representation, not architecture.

## Usage

### Configuration

In `hyperparams_config.json`:

```json
{
  "network_type": "boolean",
  "comment_network_type": "Options: 'boolean', 'embedding', 'multi_encoder'",
  "embedding_dim": 52,
  "comment_embedding_dim": "For embedding-based network only"
}
```

### Code Example

```python
from cuttle.networks import NeuralNetwork, EmbeddingBasedNetwork, MultiEncoderNetwork

# Boolean network (baseline)
model = NeuralNetwork(observation_space, embedding_size, num_actions, None)

# Embedding-based network
model = EmbeddingBasedNetwork(observation_space, embedding_dim=52, num_actions=num_actions)

# Multi-encoder network
model = MultiEncoderNetwork(observation_space, num_actions=num_actions)
```

## Implementation Details

### Observation Compatibility

- **All networks accept same observation dict**: No changes to `CuttleEnvironment.get_obs()`
- **Different preprocessing**: Each network converts observations to its preferred format internally
- **Same interface**: All networks implement `forward(observation)` → Q-values

### Model Incompatibility

- **Different architectures**: Cannot share weights between network types
- **Separate training**: Each network type trains independently
- **Same evaluation**: All can be evaluated using same metrics/validation

## Related Concepts

- [[Neural Network Basics]] - Fundamental neural network concepts
- [[Deep Q-Network]] - DQN algorithm using these networks
- [[Observation Space]] - Input format for all networks
- [[Reward Engineering]] - Rewards used during training
- [[Hyperparameters]] - Configuration options

---
*Network architectures for Cuttle DQN project*
