---
title: Project Overview
tags: [project, overview, cuttle]
created: 2026-01-16
related: [CuttleEnvironment, Neural Network Architecture, Training Process]
---

# Project Overview

## Summary

A reinforcement learning project training Deep Q-Network (DQN) agents to play the **Cuttle** card game through [[Self-Play]] training. The agent learns optimal strategies by competing against itself and other heuristic opponents.

## Key Components

### Core Modules

1. **[[CuttleEnvironment]]** - Gymnasium-compatible game environment
   - Manages game state, rules, and transitions
   - Provides uniform boolean array observations
   - Handles action execution and rewards

2. **[[Network Architectures]]** - Multiple DQN network implementations
   - Three network types: Boolean, Embedding-Based, Multi-Encoder
   - All share 52-neuron game-based hidden layer
   - Different preprocessing strategies for input representation
   - See [[Input Representation Experiments]] for experimental design

3. **[[Training Process]]** - Self-play training system
   - Agents train by playing against themselves
   - Uses [[Experience Replay]] buffer
   - [[Epsilon-Greedy Exploration|Epsilon-greedy]] exploration strategy

4. **[[Action System]]** - Action registry and execution
   - Manages all possible game actions with integer indices
   - Handles complex card game mechanics

### Key Features

- **Observation Space**: Uniform boolean arrays (no embeddings)
  - 7 zones × 52 cards = 364 booleans
  - Stack: 52 booleans
  - Effect-shown: 52 booleans
  - **Total: 468 boolean features**

- **Action Space**: Discrete actions via [[Action System|ActionRegistry]]
  - Integer indices for all valid game actions

- **Training Methodology**: [[Self-Play]] with multiple opponents
  - Training rounds with episodes per round
  - Validation against heuristic opponents ([[Randomized Player|Randomized]], [[Score Gap Maximizer|GapMaximizer]])
  - [[Early Stopping]] to prevent overfitting

## Project Structure

```
src/cuttle/
├── environment.py   # CuttleEnvironment
├── networks.py      # NeuralNetwork (DQN)
├── training.py      # selfPlayTraining
├── actions.py       # ActionRegistry
└── players.py       # Agent, Randomized, HeuristicHighCard, etc.
```

## Technologies Used

- **[[PyTorch]]** - Deep learning framework for neural networks
- **[[Gymnasium]]** - RL environment interface (formerly OpenAI Gym)
- **[[NumPy]]** - Numerical computing for game state representation
- **[[Matplotlib & Seaborn]]** - Visualization of training metrics

## Training Configuration

Configured via `hyperparams_config.json`:
- Learning rate: 3e-5
- Gamma (discount factor): 0.90
- Batch size: 128
- Replay buffer size: 30,000
- Epsilon decay: 28,510 episodes
- Target network update frequency: 500 steps

See [[Hyperparameters]] for detailed explanation.

## Research Focus

This project explores:
1. **[[Self-Play]]** training effectiveness for card games
2. **[[Reward Engineering]]** for complex multi-turn games
3. **[[Deep Q-Network|DQN]]** scalability to card game state spaces
4. **[[Input Representation Experiments]]** - Comparing preprocessing strategies:
   - Boolean concatenation (baseline)
   - Card embeddings with zone aggregation
   - Zone-specific encoders
5. Training stability and convergence in card game RL

## Related Concepts

- [[Reinforcement Learning]]
- [[Deep Q-Network]]
- [[Self-Play]]
- [[Experience Replay]]
- [[Q-Learning]]

---
*Part of the Cuttle DQN training project*
