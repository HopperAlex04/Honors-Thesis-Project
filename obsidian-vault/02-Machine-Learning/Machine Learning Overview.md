---
title: Machine Learning Overview
tags: [machine-learning, fundamentals, theory]
created: 2026-01-16
related: [Supervised Learning, Unsupervised Learning, Reinforcement Learning]
---

# Machine Learning Overview

## Definition

**Machine Learning (ML)** is a subset of artificial intelligence where algorithms learn patterns from data without being explicitly programmed for every scenario. Instead of hard-coding rules, the system discovers patterns through experience.

## Core Paradigms

### 1. [[Supervised Learning]]

Learning from labeled examples.

- **Input**: Labeled training data `(X, y)` where `X` are features and `y` are labels
- **Goal**: Learn function `f: X → y` that generalizes to unseen data
- **Examples**: 
  - Classification: Spam detection (emails → spam/not spam)
  - Regression: House price prediction (features → price)
- **Training**: Minimize loss between predictions and true labels

### 2. [[Unsupervised Learning]]

Discovering patterns in unlabeled data.

- **Input**: Unlabeled data `X` (no target labels)
- **Goal**: Find hidden structure, patterns, or representations
- **Examples**:
  - Clustering: Group similar data points
  - Dimensionality reduction: Find lower-dimensional representations
  - Anomaly detection: Identify outliers
- **Training**: Optimize objective function (e.g., reconstruction error)

### 3. [[Reinforcement Learning]]

Learning through interaction and feedback.

- **Input**: Agent interacts with environment
- **Goal**: Learn optimal policy to maximize cumulative reward
- **Process**: Agent takes actions → receives rewards → updates policy
- **Examples**: 
  - Game playing (chess, Go, card games)
  - Robotics
  - Autonomous driving
- **Training**: Trial and error with reward signals

## Key Concepts

### Training vs Testing

- **Training Set**: Data used to learn the model
- **Validation Set**: Data used to tune hyperparameters and select models
- **Test Set**: Data used for final evaluation (never seen during training)

### Generalization

The ability to perform well on unseen data:
- **Overfitting**: Model memorizes training data, performs poorly on test data
- **Underfitting**: Model too simple, fails to capture patterns
- **Regularization**: Techniques to prevent overfitting (see [[Regularization]])

### Loss Functions

Measures how wrong predictions are:
- **Classification**: Cross-entropy, Hinge loss
- **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- See [[Loss Functions]] for details

### Optimization

Finding model parameters that minimize loss:
- **Gradient Descent**: Iteratively update parameters in direction of steepest descent
- **Learning Rate**: Step size in parameter updates
- See [[Optimization Algorithms]] for variants

## ML Workflow

1. **Data Collection** - Gather relevant data
2. **Preprocessing** - Clean, normalize, feature engineering
3. **Model Selection** - Choose appropriate algorithm
4. **Training** - Learn parameters from data
5. **Validation** - Tune hyperparameters
6. **Evaluation** - Test on held-out data
7. **Deployment** - Use model for predictions

## ML in This Project

This project uses **[[Reinforcement Learning]]**, specifically:
- **[[Deep Q-Network]]** for value function approximation
- **[[Self-Play]]** for generating training data
- **[[Experience Replay]]** for learning from past experiences
- [[Neural Network Architecture|Neural networks]] as function approximators

## Related Concepts

- [[Supervised Learning]] - Learning from labels
- [[Unsupervised Learning]] - Pattern discovery
- [[Reinforcement Learning]] - Learning through interaction
- [[Neural Network Basics]] - Function approximators
- [[Deep Q-Network]] - Application in this project

## Further Reading

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

---
*Fundamental ML concepts for thesis reference*
