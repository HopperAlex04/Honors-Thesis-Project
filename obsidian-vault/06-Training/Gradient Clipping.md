---
title: Gradient Clipping
tags: [training, optimization, neural-networks]
created: 2026-01-16
related: [Backpropagation, Optimization Algorithms, Hyperparameters]
---

# Gradient Clipping

## Overview

**Gradient Clipping** is a technique used during neural network training to prevent gradients from becoming too large (exploding gradients). It limits the magnitude of gradients by clipping them to a maximum value, stabilizing training.

## The Problem: Exploding Gradients

### What Are Exploding Gradients?

Gradients that grow exponentially during [[Backpropagation]]:
- Can cause weights to update by huge amounts
- Leads to unstable training
- Can cause loss to become NaN (Not a Number)
- Network may fail to converge

### Why Do Gradients Explode?

1. **Large weight values**: Weights accumulate over training
2. **Deep networks**: Gradients multiply through many layers
3. **Activation functions**: Some activations can amplify gradients
4. **Large learning rates**: Combined with large gradients → huge updates

### Effect

```
Large gradients → Large weight updates → Unstable network → Training fails
```

## Solution: Gradient Clipping

### Concept

Limit gradient magnitude to maximum threshold:
- If gradient norm > threshold: Scale down to threshold
- If gradient norm ≤ threshold: Keep unchanged

### Methods

#### 1. Gradient Norm Clipping (L2 Norm)

Clip the gradient vector to maximum L2 norm:

```python
# Compute gradient norm
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Process**:
1. Compute L2 norm of all gradients: `||g|| = sqrt(Σ g²)`
2. If `||g|| > max_norm`: Scale gradients: `g' = g * (max_norm / ||g||)`
3. If `||g|| ≤ max_norm`: Keep gradients unchanged

**This project**: Uses this method with `max_norm = 5.0`

#### 2. Value Clipping

Clip individual gradient values:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Process**:
- Clip each gradient value to `[-clip_value, clip_value]`
- Less common than norm clipping

## Implementation in PyTorch

### Gradient Norm Clipping

```python
import torch.nn.utils as utils

# Compute loss and backpropagate
loss.backward()

# Clip gradients
utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Update weights
optimizer.step()
```

**Returns**: Gradient norm before clipping (useful for monitoring)

### Gradient Value Clipping

```python
# Clip individual gradient values
utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

## Choosing the Threshold

### Gradient Norm Threshold

**Typical values**: 1.0 - 10.0

- **Too small** (e.g., 0.1): Clips too aggressively, slows learning
- **Too large** (e.g., 100.0): Doesn't prevent explosions
- **Common**: 1.0 - 5.0

**This project**: `max_norm = 5.0` (see [[Hyperparameters]])

### Monitoring

Monitor gradient norms during training:
- **Normal**: Gradient norms in reasonable range (1-10)
- **Exploding**: Gradient norms spike to hundreds/thousands
- **Vanishing**: Gradient norms very small (< 0.01)

## Benefits

### 1. Prevents Explosion

Limits gradient magnitude:
- Prevents weight updates from becoming huge
- Keeps training stable

### 2. Stabilizes Training

More consistent training:
- Loss doesn't spike unexpectedly
- Network converges more reliably

### 3. Allows Larger Learning Rates

With gradient clipping, can sometimes use larger learning rates:
- Clipping prevents explosion from large gradients
- Still need to be careful with learning rate

## Limitations

### 1. Doesn't Fix Root Cause

Gradient clipping treats symptom, not cause:
- Large gradients may indicate other problems
- Consider: Lower learning rate, better weight initialization, network architecture

### 2. Can Slow Learning

If gradients are clipped frequently:
- Updates may be smaller than optimal
- May slow convergence

### 3. Not Always Needed

Not all networks need gradient clipping:
- Simple networks: Often not needed
- Deep networks: More likely to benefit
- Recurrent networks: Very common (LSTM, GRU)

## When to Use

### Definitely Use

- **Deep networks**: Many layers (gradients multiply)
- **Recurrent networks**: LSTMs, GRUs (time dependencies)
- **Unstable training**: Loss becomes NaN, gradients spike
- **Large learning rates**: Combined with risk of explosion

### May Not Need

- **Shallow networks**: Few layers (less gradient multiplication)
- **Stable training**: Gradients already in reasonable range
- **Small learning rates**: Low risk of explosion

## Gradient Clipping in This Project

### Configuration

From `hyperparams_config.json`:
```json
"gradient_clip_norm": 5.0
```

### Implementation

Used in [[Training Process]]:
```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
```

### Rationale

- **DQN network**: 2 hidden layers (relatively shallow, but still benefits)
- **Training stability**: Prevents gradient explosions
- **Q-value stability**: Helps keep Q-values in reasonable range

## Alternative Techniques

### 1. Weight Initialization

Initialize weights carefully:
- **Xavier/Glorot**: For sigmoid/tanh
- **He initialization**: For ReLU
- Prevents initial weights from being too large

### 2. Learning Rate Scheduling

Reduce learning rate over time:
- Starts with larger learning rate
- Decays to smaller learning rate
- Reduces risk of large updates later

### 3. Batch Normalization

Normalize activations:
- Reduces internal covariate shift
- Can help with gradient flow
- Less common in RL (changes training dynamics)

## Related Concepts

- [[Backpropagation]] - Where gradients come from
- [[Optimization Algorithms]] - How gradients are used
- [[Hyperparameters]] - Gradient clipping threshold
- [[Training Process]] - Implementation in this project

## Further Reading

- Pascanu, R., et al. (2013). "On the difficulty of training recurrent neural networks." *ICML*.

---
*Gradient clipping technique for stable training*
