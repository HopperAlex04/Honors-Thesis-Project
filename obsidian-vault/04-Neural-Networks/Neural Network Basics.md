---
title: Neural Network Basics
tags: [neural-networks, machine-learning, deep-learning]
created: 2026-01-16
related: [Feedforward Networks, Activation Functions, Backpropagation, Deep Q-Network]
---

# Neural Network Basics

## Overview

**Neural Networks** are computing systems inspired by biological neurons. They consist of interconnected nodes (neurons) organized in layers, capable of learning complex patterns from data through [[Backpropagation|backpropagation]] and gradient descent.

## Biological Inspiration

### Biological Neuron

- **Dendrites**: Receive input signals
- **Cell Body**: Processes signals
- **Axon**: Sends output signals
- **Synapses**: Connections to other neurons (strengths vary)

### Artificial Neuron

- **Inputs**: x₁, x₂, ..., xₙ
- **Weights**: w₁, w₂, ..., wₙ (connection strengths)
- **Bias**: b (threshold)
- **Activation Function**: f (determines output)
- **Output**: y = f(Σ(wᵢxᵢ) + b)

## Network Structure

### Layers

Neural networks are organized in layers:

1. **Input Layer**: Receives input data
2. **Hidden Layers**: Intermediate processing layers (can be many)
3. **Output Layer**: Produces final output

### Fully Connected Layer

Each neuron in layer L is connected to all neurons in layer L+1:
- **Weights**: Matrix W (rows = neurons in L, columns = neurons in L+1)
- **Biases**: Vector b (one per neuron in L+1)
- **Operation**: y = f(Wx + b)

## Forward Propagation

Computing output from input:

```
Input: x
  ↓
Layer 1: h₁ = f₁(W₁x + b₁)
  ↓
Layer 2: h₂ = f₂(W₂h₁ + b₂)
  ↓
Layer 3: y = f₃(W₃h₂ + b₃)
  ↓
Output: y
```

Each layer applies: **linear transformation** (Wx + b) → **activation function** f(·)

See [[Feedforward Networks]] for details.

## Activation Functions

Non-linear functions that introduce non-linearity (enables learning complex patterns):

### Common Activation Functions

1. **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`
   - Most common in hidden layers
   - Used in this project's DQN
   - See [[Activation Functions]] for details

2. **Sigmoid**: `f(x) = 1/(1 + e⁻ˣ)`
   - Output: (0, 1)
   - Used for binary classification

3. **Tanh**: `f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)`
   - Output: (-1, 1)
   - Symmetric version of sigmoid

4. **Softmax**: `f(xᵢ) = eˣⁱ / Σⱼeˣʲ`
   - Output: Probability distribution
   - Used for multi-class classification

See [[Activation Functions]] for more details.

## Learning Process

### 1. Forward Pass

Compute predictions from inputs using current weights.

### 2. Compute Loss

Compare predictions to true labels using [[Loss Functions|loss function]]:
```
Loss = L(predictions, true_labels)
```

### 3. Backward Pass (Backpropagation)

Compute gradients of loss with respect to weights:
```
∇w L = ∂L/∂w
```

Propagate gradients backward through network (chain rule).

### 4. Update Weights

Update weights using gradient descent:
```
w ← w - α ∇w L
```

Where α is learning rate.

See [[Backpropagation]] for details.

## Neural Network Types

### Feedforward Networks

Information flows in one direction (input → output):
- Most common type
- Used in this project's DQN
- See [[Feedforward Networks]]

### Recurrent Neural Networks (RNNs)

Have connections that form cycles:
- Process sequences (time series, text)
- Maintain hidden state
- Examples: LSTM, GRU

### Convolutional Neural Networks (CNNs)

Use convolutional layers:
- Designed for grid-like data (images)
- Share weights (translation invariance)
- Used in computer vision

## Deep Learning

**Deep Learning** = Neural networks with many hidden layers:
- **Deep networks**: 3+ hidden layers
- Can learn hierarchical representations
- More layers = more complex patterns (but harder to train)

This project's [[Neural Network Architecture|DQN]] has 2 hidden layers (512 → 256) - relatively shallow.

## Training Neural Networks

### Gradient Descent

Optimization algorithm that minimizes loss:
- Compute gradients
- Update weights in direction of negative gradient
- Iterate until convergence

See [[Optimization Algorithms]] for variants (SGD, Adam, etc.).

### Loss Functions

Measure how wrong predictions are:
- **Regression**: Mean Squared Error (MSE)
- **Classification**: Cross-Entropy Loss
- See [[Loss Functions]] for details

### Regularization

Techniques to prevent overfitting:
- **Dropout**: Randomly disable neurons during training
- **Weight Decay / L2 Regularization**: Penalize large weights
- **Early Stopping**: Stop training when validation loss increases
- See [[Regularization]] for details

## Neural Networks in This Project

### Architecture Overview

This project implements three network architectures (see [[Network Architectures]] for details):

1. **Boolean Network**: Simple concatenation → 52 neurons → num_actions
2. **Embedding-Based Network**: Card embeddings → zone aggregation → 52 neurons → num_actions
3. **Multi-Encoder Network**: Zone encoders → fusion → 52 neurons → num_actions

### Shared Architecture

**All networks share a game-based hidden layer**:
```
Preprocessing (varies)
  → 52 neurons (game-based, one per card) + ReLU
  → num_actions (no activation, unbounded Q-values)
```

- **52-neuron hidden layer**: Game-based design (one neuron per card)
- **Output layer**: No activation (unbounded Q-values)
- **Fully connected** (dense layers)

### Purpose

Approximate Q-function Q(s,a) for [[Deep Q-Network|DQN]]:
- Input: Game state (observation dictionary)
- Output: Q-values for all actions
- **Experimental variable**: Preprocessing/input representation complexity

### Training

- Loss: Mean Squared Error (TD error)
- Optimizer: Adam (see [[Optimization Algorithms]])
- Framework: [[PyTorch]]

### Network Types

See [[Network Architectures]] for detailed descriptions of each network type and [[Input Representation Experiments]] for the experimental design.

## Key Concepts

### Universal Approximation Theorem

A neural network with sufficient capacity can approximate any continuous function arbitrarily well. This justifies using neural networks as function approximators.

### Non-Linearity

Activation functions introduce non-linearity - without them, multiple layers would be equivalent to a single layer (only linear transformations).

### Depth vs Width

- **Depth**: Number of layers (deeper = more hierarchical features)
- **Width**: Number of neurons per layer (wider = more capacity per layer)

Trade-off between depth and width depends on problem.

## Related Concepts

- [[Feedforward Networks]] - Forward propagation details
- [[Activation Functions]] - Non-linearities
- [[Backpropagation]] - Gradient computation
- [[Loss Functions]] - Measuring error
- [[Optimization Algorithms]] - Training methods
- [[Deep Q-Network]] - Application in this project

## Further Reading

- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
- Nielsen, M. (2015). *Neural Networks and Deep Learning*. Online book.

---
*Neural network fundamentals for thesis reference*
