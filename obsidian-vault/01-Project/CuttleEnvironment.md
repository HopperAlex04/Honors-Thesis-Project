---
title: CuttleEnvironment
tags: [project, environment, gymnasium]
created: 2026-01-16
related: [Project Overview, Observation Space, Action System, Gymnasium]
---

# CuttleEnvironment

## Overview

**CuttleEnvironment** is a Gymnasium-compatible game environment for the Cuttle card game. It manages game state, executes actions, and provides observations and rewards to reinforcement learning agents.

## Implementation

Located in `src/cuttle/environment.py`

### Class Definition

```python
class CuttleEnvironment:
    def __init__(self):
        # Initialize game state, observation space, action space
    
    def reset(self, seed=None, options=None):
        # Reset environment, return initial observation
    
    def step(self, action):
        # Execute action, return (observation, reward, terminated, truncated, info)
```

## Game State

### Zones

Boolean arrays representing card locations (each array length 52):

- **Player Hand**: Cards in player's hand
- **Player Field**: Cards on player's field
- **Player Revealed**: Cards in player's hand that are public
- **Dealer Hand**: Cards in dealer's hand (opponent)
- **Dealer Field**: Cards on dealer's field (opponent)
- **Dealer Revealed**: Cards in dealer's hand that are public
- **Deck**: Cards still in deck
- **Scrap**: Discarded cards

### Special Zones

- **Stack**: Cards involved in current stack (boolean array, length 52)
- **Effect-Shown**: Cards shown by effects (boolean array, length 52)

### Card Indexing

Cards indexed by: `card_index = 13 * suit + rank`
- Suit: 0-3 (Clubs, Diamonds, Hearts, Spades)
- Rank: 0-12 (Ace through King)

## Observation Space

See [[Observation Space]] for detailed structure.

**Format**: Dictionary with:
- **Current Zones**: Hand, Field, Revealed (player)
- **Off Zones**: Hand, Field, Revealed (dealer/opponent)
- **Deck**: Boolean array (52)
- **Scrap**: Boolean array (52)
- **Stack**: Boolean array (52)
- **Effect-Shown**: Boolean array (52)

**Total**: 468 boolean features (7 zones × 52 + stack + effect_shown)

## Action Space

Discrete actions via [[Action System|ActionRegistry]]:
- **Format**: `spaces.Discrete(num_actions)`
- **Actions**: Integer indices for all valid game actions
- **Managed by**: ActionRegistry in `actions.py`

## Core Methods

### `reset(seed=None, options=None)`

Reset environment to initial state:
- Initialize all zones (empty except deck)
- Deal cards to players
- Return initial observation

**Returns**: `(observation, info)`

### `step(action)`

Execute action in environment:
- Validate action
- Update game state
- Check for terminal conditions
- Compute reward
- Return next observation

**Returns**: `(observation, reward, terminated, truncated, info)`

**Parameters**:
- `action`: Integer action index

**Returns**:
- `observation`: Next game state (dict)
- `reward`: Reward for this step (float)
- `terminated`: Whether episode ended naturally (bool)
- `truncated`: Whether episode ended due to limit (bool)
- `info`: Additional debugging info (dict)

## Rewards

See [[Reward Engineering]] for detailed reward structure.

### Terminal Rewards

- **Win**: +1.0
- **Loss**: -1.0
- **Draw**: -0.5

### Intermediate Rewards

- **Score changes**: `0.01 * score_change`
- **Gap changes**: `0.005 * gap_change`

## Game Logic

### Turn Management

- Tracks current player (player/dealer)
- Handles turn transitions
- Manages action sequences (stacks, counters)

### Action Execution

- Validates actions via ActionRegistry
- Updates zones based on action type
- Handles special card effects
- Manages stack resolution

### Terminal Conditions

Game ends when:
- One player reaches target score (typically 21)
- Both players pass
- Maximum turns reached (safety limit)

## Integration with Gymnasium

### Gymnasium API

Implements standard Gymnasium interface:
- `observation_space`: Dict space with boolean arrays
- `action_space`: Discrete space
- `reset()`: Reset method
- `step()`: Step method

### Compatibility

Works with any Gymnasium-compatible RL library:
- Stable-Baselines3
- Custom implementations (like this project)

See [[Gymnasium]] for environment interface details.

## Usage Example

```python
from cuttle.environment import CuttleEnvironment

# Create environment
env = CuttleEnvironment()

# Reset
observation, info = env.reset()

# Play game
done = False
while not done:
    action = agent.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Related Concepts

- [[Project Overview]] - Project context
- [[Observation Space]] - Observation structure
- [[Action System]] - Action management
- [[Gymnasium]] - Environment interface
- [[Reward Engineering]] - Reward structure
- [[Training Process]] - Used in training

---
*Cuttle game environment implementation*
