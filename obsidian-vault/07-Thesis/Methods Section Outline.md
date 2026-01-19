---
title: Methods Section Outline
tags: [thesis, methods, outline, research-methodology]
created: 2026-01-19
related: [Research Considerations, Input Representation Experiments, Statistical Significance and Multiple Runs, Reward Engineering]
---

# Methods Section Outline

## Overview

This document provides a structured outline for the Methods section of the thesis, with academic sources that support each methodological choice. The Methods section should demonstrate rigorous experimental design and justify each decision.

---

## 1. Game Environment

### 1.1 Cuttle Card Game

**Content**: Description of Cuttle as the experimental domain
- Two-player competitive card game
- Perfect information (all cards visible once played)
- Complex action space (~3,157 possible actions)
- Variable-length episodes (10-50+ turns)

**Justification**: Card games provide challenging RL testbeds due to combinatorial state spaces, strategic depth, and sequential decision-making.

**Sources**:
- Zha, D., et al. (2019). "RLCard: A Toolkit for Reinforcement Learning in Card Games." *arXiv:1910.04376*. [Establishes card games as valid RL benchmarks]
- Heinrich, J., & Silver, D. (2016). "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games." *arXiv:1603.01121*. [DRL applied to card games]

### 1.2 Environment Implementation

**Content**: Gymnasium-compatible environment
- State representation: Boolean arrays (468 features)
- Action space: Discrete (ActionRegistry with ~3,157 actions)
- Reward structure: Terminal and intermediate rewards

**Sources**:
- Brockman, G., et al. (2016). "OpenAI Gym." *arXiv:1606.01540*. [Standard RL environment interface]
- Towers, M., et al. (2023). "Gymnasium: A Standard Interface for Reinforcement Learning Environments." *arXiv:2407.17032*. [Updated Gym specification]

---

## 2. Neural Network Architectures

### 2.1 Deep Q-Network (DQN) Framework

**Content**: Core algorithm description
- Q-learning with function approximation
- Experience replay buffer (30,000 transitions)
- Target network with soft updates (τ = 0.005)
- Epsilon-greedy exploration (ε: 0.9 → 0.05)

**Sources**:
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533. [Foundational DQN paper]
- van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*. [Double DQN improvements]

### 2.2 Input Representation Comparison (Independent Variable)

**Content**: Three preprocessing strategies compared
1. **Boolean Network**: Direct concatenation of boolean features
2. **Embedding-Based Network**: Learned card embeddings with zone aggregation
3. **Multi-Encoder Network**: Zone-specific encoders with fusion

**Justification**: Input representation significantly affects learning efficiency in deep RL; comparing strategies isolates this effect.

**Sources**:
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NeurIPS*. [Embedding representations]
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*. [Multi-head/multi-encoder architectures]
- Dulac-Arnold, G., et al. (2015). "Deep Reinforcement Learning in Large Discrete Action Spaces." *arXiv:1512.07679*. [Action embedding for large action spaces]

### 2.3 Shared Architecture Components (Controlled Variables)

**Content**: Constant across all network types
- 52-neuron game-based hidden layer
- ReLU activation functions
- Same output layer dimension
- Same parameter initialization scheme

**Justification**: Controlling architecture ensures performance differences are attributable to input representation, not network capacity.

**Sources**:
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6: Feedforward Networks. [Architecture design principles]
- He, K., et al. (2015). "Delving Deep into Rectifiers." *ICCV*. [Weight initialization]

---

## 3. Training Methodology

### 3.1 Self-Play Training

**Content**: Agent trains by playing against itself
- Both players controlled by same network (alternating roles)
- No external opponent during training phase
- Natural curriculum as agent improves

**Justification**: Self-play enables learning without expert demonstrations and provides automatic curriculum.

**Sources**:
- Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." *Nature*, 550(7676), 354-359. [AlphaGo Zero self-play]
- Silver, D., et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science*, 362(6419), 1140-1144. [AlphaZero generalization]
- Lanctot, M., et al. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." *NeurIPS*. [Theoretical foundations of self-play]

### 3.2 Reward Structure

**Content**: Two experimental conditions
1. **Sparse Rewards**: Terminal only (Win: +1, Loss: -1, Draw: 0)
2. **Intermediate Rewards**: Terminal + score-based (0.01 per point) + gap-based (0.005 per point)

**Justification**: Reward shaping addresses credit assignment in long-horizon tasks; comparing sparse vs. shaped rewards demonstrates necessity of intermediate signals. (Maybe in this section demonstrate the data showing why the extra rewards were necessary for more competent and focused play)

**Sources**:
- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML*. [Theoretical foundation for reward shaping]
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 15: Temporal Difference Learning. [Credit assignment problem]
- Arjona-Medina, J. A., et al. (2019). "RUDDER: Return Decomposition for Delayed Rewards." *NeurIPS*. [Credit assignment in delayed reward settings]

### 3.3 Training Schedule

**Content**: Round-based training structure
- 20 rounds × 250 episodes per round = 5000 total training episodes
- Validation after each round against fixed opponents
- Learning rate decay: 0.9× every 5 rounds

**Sources**:
- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." *WACV*. [Learning rate scheduling]

---

## 4. Evaluation Methodology

### 4.1 Validation Opponents

**Content**: Fixed opponents for consistent evaluation
1. **Randomized**: Uniform random action selection (baseline)
2. **GapMaximizer**: Heuristic that maximizes score differential

**Justification**: Fixed opponents provide stable benchmarks unaffected by training dynamics.

**Sources**:
- Machado, M. C., et al. (2018). "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." *JAIR*. [Evaluation methodology in RL]

### 4.2 Performance Metrics

**Content**: Primary and secondary metrics
- **Primary**: Win rate against validation opponents
- **Secondary**: Training loss, Q-value stability, convergence speed

**Sources**:
- Henderson, P., et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI*. [Evaluation best practices in deep RL]

---

## 5. Statistical Methodology

### 5.1 Multiple Independent Runs

**Content**: 7 runs per network type with different random seeds
- Total: 21 runs (3 network types × 7 runs)
- Each run uses unique seed for reproducibility
- Same hyperparameters across all runs

**Justification**: RL training has high variance; multiple runs required for statistical validity.

**Sources**:
- Colas, C., et al. (2019). "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments." *arXiv:1806.08295*. [Sample size justification for RL experiments]
- Henderson, P., et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI*. [Variance in RL training]

### 5.2 Statistical Tests

**Content**: Analysis methodology
- **Normality**: Shapiro-Wilk test
- **Variance**: Levene's test
- **Comparison**: Independent t-test (if normal) or Mann-Whitney U (if non-normal)
- **Multiple comparisons**: Bonferroni correction (α = 0.05/3 = 0.0167)
- **Effect size**: Cohen's d for pairwise comparisons

**Sources**:
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum. [Effect size interpretation]
- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*. [Multiple comparison procedures in ML]

### 5.3 Confidence Intervals

**Content**: 95% confidence intervals for all reported metrics
- Formula: mean ± t_(α/2, n-1) × (std / √n)
- For n=7: t ≈ 2.447

**Sources**:
- Cumming, G. (2014). "The New Statistics: Why and How." *Psychological Science*. [CI reporting best practices]

---

## 6. Implementation Details

### 6.1 Software Stack

**Content**: Technologies used
- PyTorch 2.x for neural networks
- Gymnasium for environment interface
- NumPy for numerical operations
- Python 3.12

**Sources**:
- Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*. [PyTorch framework]

### 6.2 Hyperparameters

**Content**: Complete hyperparameter specification
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-4 | Standard for DQN |
| Batch size | 128 | Balance of stability and speed |
| Discount factor (γ) | 0.9 | Moderate future reward weighting |
| Replay buffer | 30,000 | ~2-3 rounds of experience |
| Target update (τ) | 0.005 | Soft update for stability |
| Gradient clip | 5.0 | Prevent gradient explosion |

**Sources**:
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*. [DQN hyperparameter baselines]

### 6.3 Reproducibility

**Content**: Measures for reproducibility
- Fixed random seeds per run
- Git commit hash recorded
- Full configuration saved with each run
- Code and data available in repository

**Sources**:
- Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*. [ML reproducibility checklist]

---

## 7. Limitations and Threats to Validity

### 7.1 Internal Validity

**Content**: Controlled experimental design
- Same hyperparameters across conditions
- Same training procedure
- Multiple runs with statistical analysis

### 7.2 External Validity

**Content**: Generalizability considerations
- Results specific to Cuttle card game
- May not generalize to all card games or game types
- Self-play training may behave differently with other training paradigms

### 7.3 Construct Validity

**Content**: Measurement validity
- Win rate against fixed opponents may not capture all aspects of "good play"
- Heuristic opponents may have exploitable weaknesses

**Sources**:
- Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and Quasi-Experimental Designs for Generalized Causal Inference*. Houghton Mifflin. [Validity framework]

---

## Summary of Key Sources

### Foundational RL/DQN
1. Mnih et al. (2015) - DQN Nature paper
2. Sutton & Barto (2018) - RL textbook
3. Silver et al. (2017, 2018) - AlphaGo/AlphaZero self-play

### Methodology
4. Henderson et al. (2018) - Deep RL evaluation practices
5. Colas et al. (2019) - Statistical power in RL
6. Ng et al. (1999) - Reward shaping theory

### Implementation
7. Paszke et al. (2019) - PyTorch
8. Brockman et al. (2016) - OpenAI Gym
9. Pineau et al. (2021) - Reproducibility

### Statistics
10. Cohen (1988) - Effect sizes
11. Demšar (2006) - ML statistical comparisons

---

## Related Documents

- [[Research Considerations]] - Framing and limitations
- [[Input Representation Experiments]] - Detailed experimental design
- [[Statistical Significance and Multiple Runs]] - Statistical methodology details
- [[Reward Engineering]] - Reward structure rationale
- [[Self-Play]] - Training methodology
- [[Hyperparameters]] - Configuration details

---
*Thesis Methods section outline with supporting literature*
