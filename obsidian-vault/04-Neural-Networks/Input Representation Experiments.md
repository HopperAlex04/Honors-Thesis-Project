---
title: Input Representation Experiments
tags: [neural-networks, experiments, input-representation, research]
created: 2026-01-16
related: [Network Architectures, Observation Space, Neural Network Basics]
---

# Input Representation Experiments

## Overview

This project conducts experiments comparing different input representation strategies for DQN agents in the Cuttle card game. The experimental design isolates **preprocessing complexity** as the variable while keeping the network architecture constant.

## Experimental Design

### Research Question

**How does input representation complexity affect DQN learning performance in card games?**

### Hypothesis

More sophisticated input representations (embeddings, zone-specific encoders) may:
- Improve learning efficiency
- Enable better credit assignment
- Capture card relationships and synergies
- Lead to better final performance

### Experimental Variable: Preprocessing Complexity

Three levels of preprocessing complexity:

1. **Boolean (Baseline)**: Simple concatenation of boolean arrays
2. **Embedding**: Learned card embeddings with zone aggregation
3. **Multi-Encoder**: Separate encoders for each zone type

### Controlled Variables

**Shared Architecture**:
- All networks use 52-neuron game-based hidden layer
- Same output layer (52 → num_actions)
- Same activation functions (ReLU)
- Same training procedure

**Same Observations**:
- All networks receive identical observation dictionaries
- No changes to environment or observation format
- Fair comparison of preprocessing strategies

## Network Architectures

See [[Network Architectures]] for detailed architecture descriptions.

### Boolean Network

**Preprocessing**: Concatenation
- Input: 468-dim boolean vector
- Complexity: Low
- Parameters: ~240K

**Rationale**: Baseline - simplest possible representation

### Embedding-Based Network

**Preprocessing**: Card embeddings + zone aggregation
- Input: Variable (default: 468-dim after fusion)
- Complexity: Medium
- Parameters: ~250K

**Rationale**: Learn card relationships and similarities

### Multi-Encoder Network

**Preprocessing**: Zone-specific encoders
- Input: Variable (default: 288-dim after fusion)
- Complexity: High
- Parameters: ~200K

**Rationale**: Zone-specific feature learning

## Experimental Methodology

### Training Protocol

1. **Same Training Procedure**: All networks trained identically
   - Same hyperparameters (learning rate, batch size, etc.)
   - Same reward structure
   - Same exploration strategy (epsilon-greedy)
   - Same validation opponents

2. **Independent Training**: Each network type trains separately
   - Cannot share weights (different architectures)
   - Fresh initialization for each
   - Same number of training episodes

3. **Evaluation Metrics**:
   - Win rate against fixed opponents
   - Training loss curves
   - Sample efficiency (episodes to reach performance threshold)
   - Final performance level

### Fair Comparison

**Ensured by**:
- Shared hidden layer architecture (52 neurons)
- Same observation format
- Same training procedure
- Same evaluation protocol

**Performance differences** → Due to input representation, not architecture

## Expected Outcomes

### Boolean Network (Baseline)

- **Advantages**: Simple, interpretable, fast
- **Disadvantages**: No learned relationships, sparse representation
- **Expected**: Baseline performance

### Embedding-Based Network

- **Advantages**: Learns card relationships, captures synergies
- **Disadvantages**: More parameters, requires more data
- **Expected**: Better learning efficiency, potentially better final performance

### Multi-Encoder Network

- **Advantages**: Zone-specific learning, modular design
- **Disadvantages**: More complex, potentially overfitting
- **Expected**: Better credit assignment, zone-specific patterns

## Analysis Plan

### Quantitative Metrics

1. **Win Rate**: Performance against fixed opponents
2. **Training Efficiency**: Episodes to reach performance threshold
3. **Final Performance**: Maximum win rate achieved
4. **Loss Curves**: Learning dynamics comparison

### Qualitative Analysis

1. **Learned Representations**: What do embeddings capture?
2. **Zone Patterns**: What do zone encoders learn?
3. **Failure Modes**: Where does each approach struggle?

## Implementation

### Configuration

Set network type in `hyperparams_config.json`:

```json
{
  "network_type": "boolean",  // or "embedding" or "multi_encoder"
  "embedding_dim": 52
}
```

### Training

All networks use the same training script (`train.py`):
- Network type selected via configuration
- Same training loop
- Same logging and checkpointing

### Evaluation

All networks evaluated identically:
- Same validation opponents
- Same evaluation protocol
- Same metrics

## Statistical Significance

For thesis-level research, each network type should be trained **multiple times** (recommended: 7 runs per type) with different random seeds to demonstrate statistical significance.

See [[Statistical Significance and Multiple Runs]] for:
- Why multiple runs are needed
- How many runs to use
- Data management strategy
- Statistical analysis methods

## Related Concepts

- [[Network Architectures]] - Detailed architecture descriptions
- [[Observation Space]] - Input format
- [[Neural Network Basics]] - Fundamental concepts
- [[Deep Q-Network]] - DQN algorithm
- [[Reward Engineering]] - Reward structure
- [[Statistical Significance and Multiple Runs]] - Running multiple experiments

---
*Experimental design for input representation comparison*
