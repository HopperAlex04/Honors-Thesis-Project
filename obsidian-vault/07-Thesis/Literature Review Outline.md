---
title: Literature Review Outline
tags: [thesis, literature-review, outline]
created: 2026-01-31
related: [Methods Section Outline, Research Considerations, Project Overview]
---

# Literature Review Outline

## Purpose

The literature review establishes the theoretical foundation for applying deep reinforcement learning (DQN) to the Cuttle card game, with particular emphasis on input representation, self-play training, and the challenge of designing learning systems without expert data. It situates this thesis within the broader fields of game AI, representation learning, and reinforcement learning methodology.

---

## I. Reinforcement Learning Foundations

### A. Markov Decision Processes and Value-Based Learning
- Markov decision process (MDP) formulation
- Value functions and Bellman equations
- Temporal difference learning
- *Key sources: Sutton & Barto (2018)*

### B. Q-Learning
- Tabular Q-learning
- Convergence properties
- *Key sources: Watkins & Dayan (1992)*

### C. Function Approximation
- Need for generalization in large state spaces
- Approximation error and stability
- *Key sources: Sutton & Barto (2018), Mnih et al. (2015)*

---

## II. Deep Reinforcement Learning and DQN

### A. The DQN Algorithm
- Q-learning with neural network function approximators
- Experience replay buffer
- Target network (fixed or soft-update)
- Epsilon-greedy exploration
- *Key sources: Mnih et al. (2015)*

### B. DQN Variants and Improvements
- Double DQN (overestimation bias)
- Dueling networks
- Prioritized experience replay (optional)
- *Key sources: van Hasselt et al. (2016), Wang et al. (2016)*

### C. Challenges in Deep RL
- Sample inefficiency
- Instability and non-stationarity
- Credit assignment in long-horizon tasks
- *Key sources: Henderson et al. (2018)*

---

## III. Reinforcement Learning for Games

### A. Board Games and Perfect Information
- Chess, Go, Shogi as RL benchmarks
- Search + learning hybrids
- *Key sources: Silver et al. (2017, 2018)*

### B. Imperfect-Information Games
- Hidden information and information sets
- Self-play in imperfect-information settings
- *Key sources: Heinrich & Silver (2016), Brown & Sandholm (2019)*

### C. Card Games as RL Benchmarks
- Combinatorial state spaces
- Strategic depth and sequential decision-making
- RLCard and related toolkits
- *Key sources: Zha et al. (2019)*

---

## IV. Self-Play Training

### A. Theoretical Foundations
- Self-play as curriculum learning
- Nash equilibrium and game-theoretic framing
- *Key sources: Lanctot et al. (2017)*

### B. Empirical Success
- AlphaGo Zero: self-play without human data
- AlphaZero: generalization across games
- *Key sources: Silver et al. (2017, 2018)*

### C. Limitations and Considerations
- Cycle detection and tie-breaking
- Opponent diversity and curriculum
- Applicability to smaller-scale domains

---

## V. State and Input Representation

### A. Representation Learning
- Learned vs. handcrafted features
- Distributed representations
- *Key sources: Mikolov et al. (2013)*

### B. Representation in Reinforcement Learning
- State representation learning
- Embeddings for discrete structures (cards, actions)
- *Key sources: Dulac-Arnold et al. (2015), Lesort et al. (2018)*

### C. Domain-Specific Representations
- Flat vs. structured input (concatenation vs. embeddings)
- Inductive biases from game structure
- *Key sources: Goodfellow et al. (2016), Vaswani et al. (2017)*

### D. Representation and Learnability
- How representation affects convergence
- When flat representations fail
- *Bridge to thesis contribution*

---

## VI. Reward Shaping and Credit Assignment

### A. The Credit Assignment Problem
- Sparse vs. dense rewards
- Long-horizon tasks and delayed feedback
- *Key sources: Sutton & Barto (2018), Arjona-Medina et al. (2019)*

### B. Reward Shaping Theory
- Policy invariance under reward transformation
- Potential-based shaping
- *Key sources: Ng et al. (1999)*

### C. Practical Reward Design
- Terminal vs. intermediate rewards
- Score-based and gap-based shaping
- Trade-offs in reward design

---

## VII. Evaluation and Reproducibility in Deep RL

### A. Variance and Statistical Rigor
- High variance across random seeds
- Need for multiple runs
- *Key sources: Henderson et al. (2018), Colas et al. (2019)*

### B. Evaluation Methodology
- Fixed vs. self-play evaluation
- Benchmark opponents and metrics
- *Key sources: Machado et al. (2018)*

### C. Reproducibility Practices
- Reproducibility crisis in ML
- Checklists and best practices
- *Key sources: Pineau et al. (2021)*

### D. Statistical Analysis
- Effect sizes (Cohen's d)
- Multiple comparison correction
- Confidence intervals
- *Key sources: Cohen (1988), Demšar (2006), Cumming (2014)*

---

## VIII. Design Without Expert Data: Research Gap

### A. Games with Professional Scenes
- AlphaGo/AlphaZero and expert bootstrapping
- Elo ratings and established benchmarks

### B. Games Without Expert Guidance
- No professional meta or optimal strategies
- No established heuristics or evaluation functions
- *Key sources: Research Considerations*

### C. Game-Structure-Based Design
- Using domain structure (52 cards, zones, ranks) for architecture
- Inductive biases when empirical guidance is unavailable
- *Bridge to thesis methodology and contribution*

---

## IX. Synthesis and Thesis Positioning

### A. Summary of Key Themes
- DQN and its applicability to card games
- Self-play as training paradigm
- Input representation as critical design choice
- Reward design for complex games
- Rigorous evaluation despite high variance

### B. Research Gap Statement
- Lack of work on Cuttle specifically
- General challenge: designing RL systems for games without expert data
- Role of input representation in learnability (not just efficiency)

### C. Thesis Contribution Preview
- Empirical comparison of boolean vs. embedding vs. multi-encoder representations
- Game-structure-based architecture design
- Findings on when representation enables vs. prevents learning

---

## Source Quick Reference

| Section | Primary Sources |
|---------|-----------------|
| I | Sutton & Barto (2018), Watkins & Dayan (1992) |
| II | Mnih et al. (2015), van Hasselt et al. (2016), Henderson et al. (2018) |
| III | Silver et al. (2017, 2018), Heinrich & Silver (2016), Zha et al. (2019) |
| IV | Lanctot et al. (2017), Silver et al. (2017, 2018) |
| V | Mikolov et al. (2013), Dulac-Arnold et al. (2015), Lesort et al. (2018) |
| VI | Ng et al. (1999), Sutton & Barto (2018), Arjona-Medina et al. (2019) |
| VII | Henderson et al. (2018), Colas et al. (2019), Pineau et al. (2021), Cohen (1988) |
| VIII | Research Considerations, Silver et al. |
| IX | Synthesis of above |

---

## Related Documents

- [[Methods Section Outline]] - Methodological choices with supporting sources
- [[Research Considerations]] - Thesis framing and limitations
- [[Project Overview]] - Project context and components

---
*Literature review outline for Cuttle DQN thesis*
