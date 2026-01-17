---
title: Optimization Algorithms
tags: [neural-networks, optimization, training]
created: 2026-01-16
related: [Neural Network Basics, Backpropagation, Gradient Clipping]
---

# Optimization Algorithms

## Overview

**Optimization Algorithms** (optimizers) are methods for updating neural network weights to minimize loss. They use gradients from [[Backpropagation]] to determine how to adjust weights in each training step.

## The Optimization Problem

### Goal

Find weights θ that minimize loss function:
```
minimize L(θ)
```

Where L(θ) is the loss (e.g., Mean Squared Error).

### Gradient Descent

Use gradients to find minimum:
```
θ ← θ - α ∇L(θ)
```

Where:
- **α (alpha)**: Learning rate (step size)
- **∇L(θ)**: Gradient of loss with respect to weights

## Common Optimizers

### 1. Stochastic Gradient Descent (SGD)

**Update rule**:
```
θ ← θ - α ∇L(θ)
```

**Properties**:
- Simple, classic optimizer
- Updates using gradient directly
- Can be slow to converge
- Sensitive to learning rate

**Variants**:
- **Momentum**: Accumulates gradient momentum
- **Nesterov**: Momentum with look-ahead

### 2. Adam (Adaptive Moment Estimation)

**Update rule** (simplified):
```
m ← β₁m + (1 - β₁)∇L(θ)      # First moment (mean)
v ← β₂v + (1 - β₂)(∇L(θ))²   # Second moment (variance)
θ ← θ - α · m / (√v + ε)
```

**Properties**:
- **Adaptive learning rate**: Adjusts per parameter
- **Momentum-like**: Uses exponential moving average
- **Works well**: Often converges faster than SGD
- **Hyperparameters**: β₁ = 0.9, β₂ = 0.999, ε = 1e-8

**Used in**: This project (via PyTorch `optim.Adam`)

### 3. RMSprop

**Update rule**:
```
v ← βv + (1 - β)(∇L(θ))²
θ ← θ - α · ∇L(θ) / (√v + ε)
```

**Properties**:
- Adaptive learning rate per parameter
- Less memory than Adam
- Good for non-stationary problems

### 4. AdaGrad

**Update rule**:
```
v ← v + (∇L(θ))²
θ ← θ - α · ∇L(θ) / (√v + ε)
```

**Properties**:
- Adaptive learning rate
- Accumulates squared gradients
- Learning rate can decay too much

## Choosing an Optimizer

### Adam (Recommended Default)

- **Pros**: Works well for most problems, adaptive, fast convergence
- **Cons**: Slightly more memory than SGD
- **Use when**: Most scenarios (deep learning, RL)

### SGD

- **Pros**: Simple, interpretable, good generalization
- **Cons**: Requires careful learning rate tuning, slower convergence
- **Use when**: Want fine control, well-tuned hyperparameters

### This Project

Uses **Adam** optimizer (via PyTorch):
- Adaptive learning rate helps with varying gradients
- Works well for DQN training
- Less hyperparameter tuning needed

## Learning Rate

### Definition

Step size in weight updates:
```
θ ← θ - α ∇L(θ)
      ↑
  learning rate
```

### Choosing Learning Rate

**Too high**:
- Unstable training (loss oscillates, may diverge)
- Weights update too much

**Too low**:
- Slow convergence
- May get stuck in local minima

**Typical range**: 1e-5 to 1e-2

**This project**: 3e-5 (0.00003) - relatively small, stable

### Learning Rate Scheduling

Adjust learning rate during training:
- **Decay**: Reduce learning rate over time
- **Warm-up**: Start small, increase gradually
- **Cyclical**: Cycle between high and low

See [[Learning Rate Scheduling]] for details.

## Optimization in This Project

### Configuration

From `hyperparams_config.json`:
```json
"learning_rate": 3e-5,
"lr_decay_rate": 0.9,
"lr_decay_interval": 5
```

### Implementation

```python
import torch.optim as optim

optimizer = optim.Adam(
    network.parameters(),
    lr=3e-5
)

# Training step
optimizer.zero_grad()      # Clear gradients
loss.backward()            # Compute gradients
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)  # Clip
optimizer.step()           # Update weights
```

### Learning Rate Scheduling

Learning rate decays by factor 0.9 every 5 rounds:
- Helps convergence in later training
- Prevents overfitting

## Common Practices

### 1. Gradient Clipping

Clip gradients to prevent explosion:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

See [[Gradient Clipping]] for details.

### 2. Zero Gradients

Clear gradients before each step:
```python
optimizer.zero_grad()  # Important! Clear previous gradients
```

**Reason**: Gradients accumulate if not cleared.

### 3. Learning Rate Warm-up

Start with small learning rate, gradually increase:
- Helps early training stability
- Less common in RL

### 4. Learning Rate Decay

Reduce learning rate over time:
- Faster learning early (large steps)
- Fine-tuning later (small steps)

## Related Concepts

- [[Backpropagation]] - How gradients are computed
- [[Learning Rate Scheduling]] - Adjusting learning rate
- [[Gradient Clipping]] - Preventing gradient explosion
- [[Hyperparameters]] - Optimizer settings
- [[PyTorch]] - Framework with optimizer implementations

## Further Reading

- Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv*.
- Ruder, S. (2016). "An overview of gradient descent optimization algorithms." *arXiv*.

---
*Optimization algorithms for neural network training*
