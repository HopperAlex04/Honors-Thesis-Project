---
title: Activation Functions
tags: [neural-networks, activation, deep-learning]
created: 2026-01-16
related: [Neural Network Basics, Feedforward Networks, Deep Q-Network]
---

# Activation Functions

## Overview

**Activation Functions** are non-linear functions applied to neuron outputs in neural networks. They introduce non-linearity, enabling networks to learn complex patterns. Without activation functions, multiple layers would be equivalent to a single linear layer.

## Why Activation Functions?

### The Problem of Linearity

Without activation functions, neural networks are just linear transformations:
```
y = W₃(W₂(W₁x + b₁) + b₂) + b₃
  = W₃W₂W₁x + ... (linear combination)
```

Multiple layers collapse into a single linear layer!

### Solution: Non-Linearity

Activation functions add non-linearity:
```
y = f₃(W₃ · f₂(W₂ · f₁(W₁x + b₁) + b₂) + b₃)
```

Now multiple layers can represent complex, non-linear functions.

## Common Activation Functions

### 1. ReLU (Rectified Linear Unit)

**Formula**: `f(x) = max(0, x)`

**Properties**:
- **Output range**: [0, ∞)
- **Derivative**: 
  - `f'(x) = 1` if x > 0
  - `f'(x) = 0` if x < 0 (vanishing gradient for negative inputs)
- **Advantages**:
  - Simple, computationally efficient
  - Addresses vanishing gradient problem (for positive inputs)
  - Most common in deep learning
- **Disadvantages**:
  - "Dying ReLU": Neurons with always-negative inputs never update (gradient = 0)

**Used in**: This project's [[Neural Network Architecture|DQN]] (hidden layers)

### 2. Sigmoid

**Formula**: `f(x) = 1 / (1 + e⁻ˣ)`

**Properties**:
- **Output range**: (0, 1)
- **Derivative**: `f'(x) = f(x)(1 - f(x))`
- **Advantages**:
  - Smooth, differentiable
  - Output is probability-like
- **Disadvantages**:
  - Vanishing gradient problem (derivative near 0 for extreme inputs)
  - Not zero-centered (outputs always positive)

**Used in**: Binary classification (output layer)

### 3. Tanh (Hyperbolic Tangent)

**Formula**: `f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`

**Properties**:
- **Output range**: (-1, 1)
- **Derivative**: `f'(x) = 1 - f(x)²`
- **Advantages**:
  - Zero-centered (outputs can be positive or negative)
  - Smooth, differentiable
- **Disadvantages**:
  - Vanishing gradient problem (similar to sigmoid)

**Used in**: Hidden layers (less common than ReLU), recurrent networks

### 4. Softmax

**Formula**: `f(xᵢ) = eˣⁱ / Σⱼeˣʲ`

**Properties**:
- **Output range**: (0, 1), sums to 1 (probability distribution)
- **Output size**: Same as input size
- **Advantages**:
  - Outputs valid probability distribution
  - Works well for multi-class classification
- **Used in**: Output layer for classification

### 5. Linear / No Activation

**Formula**: `f(x) = x` (identity function)

**Properties**:
- **Output range**: (-∞, ∞)
- **Derivative**: `f'(x) = 1`
- **Used in**: Output layers for regression, Q-value outputs

**This project**: DQN output layer has no activation (unbounded Q-values)

## Activation Functions in This Project

### Network Architecture

From [[Neural Network Architecture|DQN]]:

```
Input (468 dims)
  → Linear(468 → 512) + ReLU      ← ReLU activation
  → Linear(512 → 256) + ReLU      ← ReLU activation
  → Linear(256 → num_actions)      ← No activation
```

### Rationale

1. **Hidden layers (ReLU)**:
   - Non-linearity enables learning complex patterns
   - Efficient computation
   - Addresses vanishing gradient (for positive inputs)

2. **Output layer (no activation)**:
   - Q-values need unbounded range (can be any real number)
   - Terminal rewards ±1.0, but Q-values can be larger
   - No need for probability-like output

## Choosing Activation Functions

### Hidden Layers

- **Default**: ReLU (most common, works well)
- **Alternative**: Tanh (zero-centered), Leaky ReLU (addresses dying ReLU)

### Output Layer

- **Regression / Q-values**: No activation (unbounded)
- **Binary classification**: Sigmoid (0-1 probability)
- **Multi-class classification**: Softmax (probability distribution)

## Derivatives (Gradients)

### ReLU

```
f'(x) = 1  if x > 0
      = 0  if x < 0
      = undefined if x = 0 (usually treated as 0)
```

**Effect**: Gradient flows through if input > 0, blocked if input < 0

### Sigmoid

```
f'(x) = f(x)(1 - f(x))
```

**Effect**: Gradient is small when input is extreme (large positive or negative)

### Tanh

```
f'(x) = 1 - f(x)²
```

**Effect**: Similar to sigmoid (vanishing gradient for extreme inputs)

## Vanishing Gradient Problem

### Problem

When activation function derivative is small (sigmoid, tanh):
- Gradients get smaller as they propagate backward
- Early layers receive tiny gradients
- Slow or no learning in early layers

### Solution

Use ReLU (or variants):
- Derivative = 1 for positive inputs
- Gradients flow through unchanged
- Enables training of deep networks

## Related Concepts

- [[Neural Network Basics]] - Role of activation functions
- [[Backpropagation]] - Gradients through activation functions
- [[Neural Network Architecture]] - Activation functions in DQN
- [[Deep Q-Network]] - Network with ReLU activations

## Further Reading

- Nair, V., & Hinton, G. E. (2010). "Rectified linear units improve restricted boltzmann machines." *ICML*.

---
*Activation functions for thesis reference*
