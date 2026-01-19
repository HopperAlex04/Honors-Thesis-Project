---
title: Research Considerations
tags: [thesis, research, methodology, limitations]
created: 2026-01-19
related: [Game-Based Architecture, Game-Based Dimension Rationale, Input Representation Experiments]
---

# Research Considerations

## Overview

This document captures important research considerations, limitations, and framing for the thesis. These points help contextualize the work and provide honest discussion of challenges faced.

## The "No Expert Data" Problem

### Games With Professional Scenes

Games like Chess and Go have significant advantages for AI research:

**AlphaGo/AlphaZero**:
- Initially trained on millions of expert human games
- Could bootstrap from centuries of documented strategy
- Clear performance benchmarks (Elo ratings, tournament results)

**Stockfish and Traditional Chess Engines**:
- Evaluation functions tuned over decades using grandmaster games
- Well-established opening books and endgame tables
- Community-validated heuristics

**Key Advantages**:
- Centuries of documented strategy and theory
- Professional players whose games can be studied
- Established evaluation functions refined over decades
- Clear performance metrics (Elo, tournament wins)

### Cuttle and Similar Games

Cuttle lacks these bootstrapping advantages:

**No Professional Scene**:
- No expert games to learn from or imitate
- No established "meta" or optimal strategies
- No Elo system to measure performance against

**No Established Heuristics**:
- What constitutes a "good" network architecture?
- How many hidden neurons are appropriate?
- What input representation captures strategic information?

**No Performance Benchmarks**:
- Is 40% win rate vs a heuristic opponent "good" or "bad"?
- No reference point for expected learning curves
- Difficult to compare to prior work (none exists for Cuttle)

### Implications for Architecture Design

Without expert guidance, we must rely on:

1. **Game Structure**: Use meaningful game units as dimension guides
   - 52 cards → 52 neurons
   - 13 ranks → 13 dimensions
   - 9 zones → 9 encoders

2. **Inductive Biases**: Build domain knowledge into the architecture
   - Card embeddings (learn card relationships)
   - Max pooling (extract dominant features per zone)
   - Zone-specific encoders (preserve game semantics)

3. **Empirical Comparison**: Test multiple approaches systematically
   - Boolean vs Embedding vs Multi-Encoder
   - Same hyperparameters for fair comparison
   - Statistical significance through multiple runs

### Thesis Framing

This can be framed as a **research contribution**:

> "In the absence of domain expertise from professional play or established literature, we adopt a game-structure-based approach to architecture design. Hidden layer dimensions are chosen to correspond to meaningful game units rather than arbitrary values. This provides interpretability and a principled rationale when empirical guidance from expert play is unavailable."

## Absolute vs Relative Performance

### What Matters for the Thesis

The research question is about **comparing input representations**, not achieving state-of-the-art Cuttle AI:

**Primary Questions**:
- Does embedding outperform boolean?
- Does multi-encoder outperform boolean?
- Why do the differences exist?

**Valid Results**:
- If all networks get 30-40% vs GapMaximizer, but embedding consistently beats boolean by 15 percentage points, that's a meaningful result
- The relative difference matters more than absolute numbers

### Why Absolute Performance May Be Limited

Several factors constrain absolute performance:

1. **Training Budget**: 5,000 episodes is modest for a complex game
2. **Self-Play Only**: No curriculum learning or opponent diversity
3. **Action Space**: ~3,157 possible actions makes learning difficult
4. **Game Complexity**: Multi-step strategies, hidden information, special effects

### Honest Framing

A thesis showing "embedding networks learn faster but plateau at 40% vs heuristic opponents" is valid if it:
- Explains why (credit assignment, state space complexity)
- Discusses limitations honestly
- Suggests future work (more training, curriculum learning, etc.)

## Reviewer Expectations

### What Thesis Reviewers Care About

For an undergraduate thesis:

1. **Clear Methodology**
   - Controlled experiment design
   - Same hyperparameters across conditions
   - Documented experimental setup

2. **Statistical Validity**
   - Multiple runs with different seeds
   - Confidence intervals on results
   - Appropriate significance tests

3. **Honest Analysis**
   - Explain what worked and what didn't
   - Acknowledge limitations
   - Don't overclaim results

4. **Insight and Understanding**
   - Why does embedding help?
   - What does this teach us about input representation?
   - How does this connect to broader RL principles?

### Negative Results Are Still Results

Partial or negative results are publishable if properly analyzed:

- "The embedding network showed 2x improvement over boolean, suggesting structured input can partially compensate for sparse rewards"
- "Despite architectural improvements, performance plateaued at 40%, indicating fundamental challenges in self-play training for complex card games"
- "These results suggest future work should explore curriculum learning or hybrid reward structures"

## Generalizable Insights

### Beyond Cuttle

The research has broader implications:

**For Game AI**:
- How to design networks when expert data is unavailable
- Trade-offs between flat and structured input representations
- Value of domain-aligned architecture choices

**For RL Applications**:
- Many real-world problems lack expert data (robotics, logistics, novel domains)
- Game-structure-based design provides a template
- Inductive biases can accelerate learning

### Contribution Statement

Possible thesis contribution framing:

> "This work demonstrates that input representation significantly impacts learning efficiency in deep reinforcement learning for complex games. Using a game-structure-based approach to architecture design, we show that embedding-based representations outperform flat boolean concatenation, even when both have equivalent parameter counts. These findings suggest that domain-aligned architectural choices can partially compensate for the absence of expert data or shaped rewards."

## Related Documents

- [[Game-Based Architecture]] - Architecture design philosophy
- [[Game-Based Dimension Rationale]] - Dimension choice rationale
- [[Input Representation Experiments]] - Experimental design
- [[Statistical Significance and Multiple Runs]] - Statistical methodology
- [[Reward Engineering]] - Reward structure decisions

---
*Research considerations and thesis framing for the Cuttle DQN project*
