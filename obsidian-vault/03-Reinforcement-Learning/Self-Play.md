---
title: Self-Play
tags: [reinforcement-learning, training, self-play]
created: 2026-01-16
related: [Deep Q-Network, Training Process, Reinforcement Learning]
---

# Self-Play

## Overview

**Self-Play** is a training methodology in reinforcement learning where an agent learns by competing against itself (or copies of itself). Instead of training against a fixed opponent, the agent faces progressively stronger opponents as it improves.

## Motivation

### Traditional RL Training

- Train against fixed opponents (heuristics, random, pre-trained)
- **Problem**: Agent overfits to specific opponent strategies
- **Problem**: Can't learn to exploit weaknesses of fixed opponents
- **Limitation**: May not reach optimal play

### Self-Play Training

- Agent plays against itself (or recent versions of itself)
- **Benefit**: Opponent strength increases as agent improves
- **Benefit**: Agent learns robust strategies (not overfit to one opponent)
- **Benefit**: Can discover novel strategies through exploration

## How It Works

### Basic Self-Play

1. Initialize agent with random weights
2. Agent plays game against itself:
   - Copy of agent plays both sides (or alternating roles)
   - Collect experiences (s, a, r, s')
3. Update agent using collected experiences
4. Repeat (agent faces improved version of itself)

### Variations

#### 1. Single Agent, Both Sides

Agent controls both players:
- Alternates roles (player/dealer in card games)
- Learns from both perspectives

#### 2. Population-Based

Maintain population of agents:
- Agent plays against random members of population
- Add new agents periodically, remove old ones
- Examples: AlphaGo (Go), OpenAI Five (Dota 2)

#### 3. Opponent Pool

Store past versions of agent:
- Agent plays against random opponent from pool
- Pool contains recent checkpoints
- Ensures diverse opponents

## Advantages

### 1. Automatic Curriculum

Opponent difficulty scales with agent ability:
- Early training: Weak opponents (random play)
- Later training: Strong opponents (learned strategies)
- **Natural curriculum learning**

### 2. Robust Strategies

Agent learns to:
- Handle various playing styles
- Not overfit to specific opponent
- Adapt to different strategies

### 3. Exploration of Strategy Space

Self-play can discover:
- Novel strategies not present in fixed opponents
- Exploitative strategies (finding weaknesses)
- Robust strategies (hard to exploit)

### 4. No Need for Expert Demonstrations

Self-play doesn't require:
- Pre-trained opponents
- Expert game knowledge
- Human demonstrations

Just needs game rules and reward signal.

## Challenges

### 1. Cycling Behavior

Agents may cycle between strategies:
- Agent A beats strategy X
- Agent B beats agent A
- Agent C beats agent B (but loses to strategy X)
- **Solution**: Opponent diversity (population, pool)

### 2. Slow Initial Learning

Early games may be random/uninformative:
- Both agents weak → random games → limited learning signal
- **Solution**: Mix with fixed opponents (e.g., randomized player)

### 3. Computational Cost

Self-play can be computationally expensive:
- Need to run games for both agents
- More games needed for convergence
- **Solution**: Efficient implementations, parallelization

### 4. Reward Shaping

Self-play may require careful [[Reward Engineering]]:
- Reward signals must encourage learning
- Intermediate rewards can guide exploration
- See [[Reward Engineering]] for details

## Self-Play in This Project

### Training Process

The project uses self-play training (see [[Training Process]]):

1. **Training Rounds**: Multiple rounds of training
2. **Episodes per Round**: Agent plays many games per round
3. **Opponent Strategy**: Agent plays against itself in self-play mode
4. **Validation**: Periodically test against fixed opponents ([[Randomized Player|Randomized]], [[Score Gap Maximizer|GapMaximizer]])

### Mixed Training

Project can also train against:
- **Randomized Player**: Random action selection
- **Score Gap Maximizer**: Heuristic that maximizes score gap
- **Self-Play**: Agent vs itself

This provides diverse training experiences.

### Implementation

- Agent learns through [[Experience Replay]] buffer
- Experiences come from self-play games
- [[Epsilon-Greedy Exploration]] ensures exploration
- Periodic validation against fixed opponents

## Famous Examples

### AlphaGo / AlphaZero

- **AlphaGo**: Defeated world champion in Go using self-play
- **AlphaZero**: Generalized to Chess, Shogi, Go
- Self-play + Monte Carlo Tree Search (MCTS)

### OpenAI Five

- Dota 2 team trained via self-play
- Population of agents playing each other
- Discovered novel strategies

### AlphaStar (StarCraft II)

- Real-time strategy game
- Self-play with population-based training
- Defeated professional players

## Related Concepts

- [[Deep Q-Network]] - Algorithm used in self-play training
- [[Training Process]] - Implementation details
- [[Reinforcement Learning]] - Broader RL context
- [[Experience Replay]] - Learning from self-play experiences
- [[Reward Engineering]] - Reward design for self-play

## Further Reading

- Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." *Nature*.
- Silver, D., et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science*.
- Vinyals, O., et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature*.

---
*Self-play training methodology used in this project*
