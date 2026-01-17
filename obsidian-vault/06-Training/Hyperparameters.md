---
title: Hyperparameters
tags: [training, hyperparameters, configuration]
created: 2026-01-16
related: [Deep Q-Network, Training Process, Learning Rate Scheduling, Gradient Clipping]
---

# Hyperparameters

## Overview

**Hyperparameters** are configuration parameters that control the learning process but are not learned by the algorithm itself. They must be set before training and significantly affect performance.

## Categories

### 1. Network Architecture

Structure of the neural network:
- Layer sizes (512, 256 in this project)
- Number of layers (2 hidden layers)
- Activation functions (ReLU)

See [[Neural Network Architecture]] for details.

### 2. Learning Parameters

Control how the agent learns:
- **Learning rate**: Step size for weight updates
- **Discount factor (γ)**: Importance of future rewards
- **Batch size**: Number of experiences per update
- **Replay buffer size**: Number of stored experiences

### 3. Exploration Parameters

Control exploration vs exploitation:
- **Epsilon (ε)**: Exploration probability
- **Epsilon decay**: How quickly exploration decreases
- **Epsilon start/end**: Initial and final exploration rates

### 4. Training Schedule

Training structure:
- **Rounds**: Number of training rounds
- **Episodes per round**: Games per round
- **Validation frequency**: How often to evaluate

### 5. Optimization Parameters

Optimization algorithm settings:
- **Optimizer**: Adam, SGD, etc.
- **Gradient clipping**: Maximum gradient norm
- **Learning rate scheduling**: How learning rate changes

## Hyperparameters in This Project

### Configuration File

Stored in `hyperparams_config.json`:

```json
{
  "embedding_size": 16,           // Deprecated (no embeddings)
  "batch_size": 128,
  "gamma": 0.90,
  "eps_start": 0.90,
  "eps_end": 0.05,
  "eps_decay": 28510,
  "tau": 0.005,
  "target_update_frequency": 500,
  "learning_rate": 3e-5,
  "lr_decay_rate": 0.9,
  "lr_decay_interval": 5,
  "gradient_clip_norm": 5.0,
  "q_value_clip": 15.0,
  "replay_buffer_size": 30000,
  ...
}
```

### Key Hyperparameters

#### Learning Rate (α)

**Value**: `3e-5` (0.00003)

- **Purpose**: Step size for weight updates
- **Effect**: 
  - Too high: Unstable learning, oscillations
  - Too low: Slow convergence
- **Typical range**: 1e-5 to 1e-3
- **Scheduling**: Decays by factor 0.9 every 5 rounds (see [[Learning Rate Scheduling]])

#### Discount Factor (γ)

**Value**: `0.90`

- **Purpose**: How much we value future rewards vs immediate rewards
- **Interpretation**: 
  - γ = 0.9 means we value next reward at 90% of immediate reward
  - γ = 0 means only care about immediate reward
- **Typical range**: 0.9 - 0.99
- **Effect**: Higher γ = more planning ahead

#### Batch Size

**Value**: `128`

- **Purpose**: Number of experiences sampled from replay buffer per update
- **Effect**:
  - Larger: More stable gradients, slower updates
  - Smaller: Faster updates, higher variance
- **Typical range**: 32 - 256

#### Replay Buffer Size

**Value**: `30,000`

- **Purpose**: Maximum number of experiences stored in replay buffer
- **Effect**:
  - Larger: More diverse experiences, more memory
  - Smaller: Less diversity, faster updates
- **Typical range**: 10,000 - 1,000,000

#### Epsilon (ε)

**Start**: `0.90`, **End**: `0.05`, **Decay**: `28,510` episodes

- **Purpose**: Exploration probability ([[Epsilon-Greedy Exploration]])
- **Schedule**: Decays from 0.90 to 0.05 over 28,510 episodes
- **Effect**: High exploration early, low exploration later

#### Target Update Frequency

**Value**: `500` steps

- **Purpose**: How often to copy main network to target network
- **Effect**: 
  - More frequent: More up-to-date targets, less stable
  - Less frequent: More stable, targets may be outdated
- **Typical range**: 100 - 10,000 steps

#### Gradient Clipping

**Value**: `5.0` (max norm)

- **Purpose**: Clip gradients to prevent explosion (see [[Gradient Clipping]])
- **Effect**: Prevents large gradients from destabilizing training
- **Method**: Clip gradient norm to maximum 5.0

#### Q-Value Clipping

**Value**: `15.0` (max absolute value)

- **Purpose**: Clip Q-values to prevent explosion
- **Effect**: Keeps Q-values in reasonable range
- **Usage**: Less common than gradient clipping

### Training Configuration

```json
{
  "training": {
    "rounds": 4,
    "eps_per_round": 500,
    "validation_opponent": "randomized"
  }
}
```

- **Rounds**: Number of training rounds
- **Episodes per round**: Games per round
- **Validation opponent**: Opponent for evaluation (randomized, gapmaximizer, or both)

### Early Stopping

```json
{
  "early_stopping": {
    "enabled": true,
    "check_interval": 50,
    "window_size": 100,
    "divergence_threshold": 0.5,
    "min_episodes": 200,
    "max_loss": 50.0
  }
}
```

- **Enabled**: Whether to use early stopping
- **Check interval**: How often to check for stopping (every 50 episodes)
- **Window size**: Episodes to consider (last 100)
- **Divergence threshold**: Loss increase threshold (0.5)
- **Min episodes**: Minimum episodes before stopping (200)
- **Max loss**: Maximum loss before stopping (50.0)

## Hyperparameter Tuning

### Manual Tuning

Trial and error:
- Start with reasonable defaults
- Adjust one hyperparameter at a time
- Monitor performance (loss, win rate, etc.)

### Grid Search

Try all combinations of hyperparameter values:
- Exhaustive but computationally expensive
- Good for small hyperparameter spaces

### Random Search

Randomly sample hyperparameter combinations:
- More efficient than grid search
- Better for large hyperparameter spaces

### Bayesian Optimization

Use probabilistic model to guide hyperparameter search:
- More efficient than random search
- Requires more implementation complexity

## Sensitivity Analysis

Some hyperparameters are more sensitive than others:

### Very Sensitive
- **Learning rate**: Small changes can significantly affect performance
- **Discount factor (γ)**: Important for long-term planning

### Moderately Sensitive
- **Batch size**: Affects stability and speed
- **Epsilon decay**: Affects exploration schedule

### Less Sensitive
- **Target update frequency**: Less critical (within reasonable range)
- **Gradient clipping threshold**: Mostly prevents issues (doesn't need fine-tuning)

## Hyperparameter Best Practices

### 1. Start with Defaults

Use known good defaults from literature (e.g., DQN papers):
- Learning rate: 1e-4 to 1e-3
- Gamma: 0.95 - 0.99
- Batch size: 64 - 256

### 2. Tune One at a Time

Change one hyperparameter, observe effect:
- Easier to understand what changed performance
- Avoid confounding effects

### 3. Monitor Metrics

Track relevant metrics:
- **Loss**: Training and validation loss
- **Win rate**: Performance against opponents
- **Q-values**: Stability and magnitude

### 4. Use Validation

Evaluate on validation set (not training set):
- Avoid overfitting to hyperparameters
- More reliable performance estimates

### 5. Document Changes

Keep track of hyperparameter changes:
- What changed, when, and why
- Helps reproduce results

## Related Concepts

- [[Deep Q-Network]] - Algorithm using these hyperparameters
- [[Training Process]] - How hyperparameters are used
- [[Learning Rate Scheduling]] - Adaptive learning rates
- [[Gradient Clipping]] - Gradient stability
- [[Epsilon-Greedy Exploration]] - Exploration hyperparameters

## Further Reading

- Bergstra, J., & Bengio, Y. (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*.

---
*Hyperparameter configuration for thesis reference*
