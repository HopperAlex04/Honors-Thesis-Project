---
title: PyTorch
tags: [technology, framework, deep-learning, pytorch]
created: 2026-01-16
related: [Neural Network Architecture, Training Process]
---

# PyTorch

## Overview

**PyTorch** is an open-source deep learning framework developed by Facebook's AI Research lab. It provides a Python interface for building and training neural networks, with automatic differentiation and GPU acceleration.

## Key Features

### 1. Dynamic Computation Graphs

Unlike static graph frameworks, PyTorch builds computation graphs dynamically:
- More flexible (can change graph structure during execution)
- Easier debugging (Python-like control flow)
- Better for research and experimentation

### 2. Automatic Differentiation

PyTorch automatically computes gradients via **autograd**:
- Tracks operations on tensors
- Builds computation graph
- Computes gradients using backpropagation
- **No manual gradient computation needed!**

### 3. GPU Acceleration

Seamless CUDA integration:
- Move tensors/computations to GPU with `.cuda()` or `.to(device)`
- Automatic CPU/GPU device management
- Efficient tensor operations on GPU

### 4. Neural Network Module (`torch.nn`)

High-level API for building neural networks:
- **Layers**: `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, etc.
- **Loss Functions**: `nn.MSELoss`, `nn.CrossEntropyLoss`, etc.
- **Optimizers**: `torch.optim.Adam`, `torch.optim.SGD`, etc.

## Core Components

### Tensors

Multi-dimensional arrays (like NumPy arrays, but with GPU support):

```python
import torch

# Create tensor
x = torch.tensor([1, 2, 3])

# Operations
y = x * 2  # Element-wise multiplication

# GPU tensor
x_gpu = x.cuda()  # or x.to(device)
```

**Difference from NumPy**:
- PyTorch tensors: `torch.tensor()`
- NumPy arrays: `np.array()`
- Can convert: `torch.from_numpy(arr)`, `tensor.numpy()`

### Autograd (Automatic Differentiation)

Tracks operations for gradient computation:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()  # Compute gradients
print(x.grad)  # 4.0 (dy/dx = 2x = 4)
```

**`requires_grad=True`**: Tells PyTorch to track gradients for this tensor.

### Neural Network Module (`torch.nn`)

#### Layers

```python
import torch.nn as nn

# Linear (fully connected) layer
linear = nn.Linear(in_features=468, out_features=512)

# Activation function
relu = nn.ReLU()

# Sequential model (stack layers)
model = nn.Sequential(
    nn.Linear(468, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_actions)
)
```

This is the structure used in this project's [[Neural Network Architecture|DQN]].

#### Loss Functions

```python
criterion = nn.MSELoss()  # Mean Squared Error

# Compute loss
loss = criterion(predictions, targets)
```

#### Optimizers

```python
import torch.optim as optim

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training step
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

## PyTorch in This Project

### Network Definition

See [[Neural Network Architecture]] for DQN implementation using `nn.Sequential`:

```python
from torch import nn

self.linear_relu_stack = nn.Sequential(
    nn.Linear(input_length, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_actions)
)
```

### Training Loop

```python
# Forward pass
q_values = network(observation)

# Compute loss (MSE of TD error)
loss = criterion(q_values, target_q_values)

# Backward pass
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)  # Gradient clipping
optimizer.step()
```

### Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
observations = observations.to(device)
```

## Common Operations

### Tensor Creation

```python
# From list
x = torch.tensor([1, 2, 3])

# Zeros, ones
x = torch.zeros(3, 4)
x = torch.ones(3, 4)

# Random
x = torch.randn(3, 4)  # Normal distribution
```

### Tensor Operations

```python
# Element-wise
y = x * 2
z = x + y

# Matrix multiplication
z = torch.matmul(x, y)  # or x @ y

# Reshape
x = x.view(-1, 468)  # Flatten to (batch, 468)
```

### Converting to/from NumPy

```python
import numpy as np

# NumPy → PyTorch
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)

# PyTorch → NumPy
tensor = torch.tensor([1, 2, 3])
arr = tensor.numpy()
```

## Best Practices

### 1. Use `torch.no_grad()` for Inference

Disable gradient tracking during inference (faster, less memory):

```python
with torch.no_grad():
    q_values = network(observation)
```

### 2. Gradient Clipping

Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

Used in this project (see [[Hyperparameters]]).

### 3. Learning Rate Scheduling

Adjust learning rate during training:

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler.step()  # Update learning rate
```

### 4. Save/Load Models

```python
# Save
torch.save(model.state_dict(), 'checkpoint.pt')

# Load
model.load_state_dict(torch.load('checkpoint.pt'))
```

## Comparison to Other Frameworks

### TensorFlow / Keras

- **TensorFlow**: Static graphs (TensorFlow 1.x), eager execution (TF 2.x)
- **Keras**: High-level API on top of TensorFlow
- **PyTorch**: Dynamic graphs, more Pythonic

### JAX

- **JAX**: Functional programming, automatic differentiation
- **PyTorch**: Imperative, object-oriented

## Related Concepts

- [[Neural Network Architecture]] - PyTorch implementation of DQN
- [[Training Process]] - Training loop using PyTorch
- [[Optimization Algorithms]] - Optimizers (Adam, SGD)
- [[Gradient Clipping]] - PyTorch gradient clipping

## Resources

- **Official Docs**: https://pytorch.org/docs/
- **Tutorials**: https://pytorch.org/tutorials/
- **PyTorch Lightning**: Higher-level framework built on PyTorch

---
*PyTorch deep learning framework used in this project*
