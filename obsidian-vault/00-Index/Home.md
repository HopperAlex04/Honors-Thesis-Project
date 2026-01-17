---
title: Cuttle DQN Thesis Notes
tags: [index, home]
created: 2026-01-16
---

# Cuttle DQN Thesis Notes

Welcome to the knowledge base for the Cuttle card game Deep Q-Network (DQN) reinforcement learning project. This vault contains organized notes for thesis writing and project reference.

## 📚 Table of Contents

### Project Documentation
- [[Project Overview]] - High-level project description and architecture
- [[CuttleEnvironment]] - Game environment implementation
- [[Neural Network Architecture]] - DQN network structure
- [[Training Process]] - Self-play training methodology
- [[Observation Space]] - Input representation and formatting
- [[Action System]] - Action registry and game actions
- [[Hyperparameters]] - Configuration and tuning

### Machine Learning Fundamentals
- [[Machine Learning Overview]] - Core ML concepts and paradigms
- [[Supervised Learning]] - Learning from labeled data
- [[Unsupervised Learning]] - Pattern discovery without labels
- [[Reinforcement Learning]] - Learning through interaction and rewards

### Reinforcement Learning Concepts
- [[Q-Learning]] - Value-based RL algorithm
- [[Deep Q-Network]] - Neural networks for Q-learning
- [[Self-Play]] - Training through agent competition
- [[Experience Replay]] - Storing and replaying past experiences
- [[Epsilon-Greedy Exploration]] - Balancing exploitation and exploration
- [[Reward Engineering]] - Designing effective reward signals
- [[Policy vs Value Functions]] - Different approaches to RL
- [[On-Policy vs Off-Policy]] - Learning algorithm categories

### Neural Networks
- [[Neural Network Basics]] - Fundamentals of artificial neural networks
- [[Feedforward Networks]] - Forward propagation
- [[Activation Functions]] - ReLU, Sigmoid, Tanh, and more
- [[Backpropagation]] - Gradient computation and weight updates
- [[Loss Functions]] - Measuring prediction error
- [[Optimization Algorithms]] - SGD, Adam, and variants
- [[Regularization]] - Preventing overfitting

### Technologies & Frameworks
- [[PyTorch]] - Deep learning framework
- [[Gymnasium]] - RL environment interface
- [[NumPy]] - Numerical computing
- [[Matplotlib & Seaborn]] - Data visualization

### Training & Optimization
- [[Hyperparameter Tuning]] - Optimizing training configuration
- [[Learning Rate Scheduling]] - Adaptive learning rates
- [[Gradient Clipping]] - Stabilizing training
- [[Early Stopping]] - Preventing overfitting
- [[Training Metrics]] - Loss, Q-values, win rates
- [[Validation]] - Evaluating model performance

## 🔗 Quick Links

### Core Concepts
- [[Deep Q-Network|DQN]] - The main algorithm used
- [[CuttleEnvironment|Game Environment]] - Gymnasium-compatible environment
- [[Self-Play Training|Self-Play]] - Training methodology

### Technical Implementation
- [[Observation Space]] - Input format (468 boolean features)
- [[Action System]] - Discrete action space
- [[Neural Network Architecture]] - 512→256→num_actions

## 📝 Note-taking Tips

- Use `[[WikiLinks]]` to link between related concepts
- Tags help organize by topic: `#machine-learning`, `#reinforcement-learning`
- Frontmatter (YAML) at the top provides metadata
- This is a living document - update as you learn!

---
*Last updated: 2026-01-16*
