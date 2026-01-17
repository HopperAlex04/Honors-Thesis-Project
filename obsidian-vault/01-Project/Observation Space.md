---
title: Observation Space
tags: [project, observation, input-representation]
created: 2026-01-16
related: [CuttleEnvironment, Neural Network Architecture]
---

# Observation Space

## Overview

The [[CuttleEnvironment]] uses a **uniform boolean array representation** for all observations. No embeddings are used - all inputs are boolean presence indicators of length 52 (one for each card in a standard deck).

## Structure

### Zones (364 booleans)

Seven zones, each represented as a boolean array of length 52:

1. **Player Hand** - Cards in player's hand
2. **Player Field** - Cards on player's field
3. **Dealer Hand** - Cards in dealer's hand (opponent)
4. **Dealer Field** - Cards on dealer's field (opponent)
5. **Player Revealed** - Cards in player's hand that are public
6. **Dealer Revealed** - Cards in dealer's hand that are public
7. **Scrap** - Discarded cards

Each zone: `np.zeros(52, dtype=bool)` where `zone[card_index] = True` if the card is present.

**Total: 7 × 52 = 364 booleans**

### Stack (52 booleans)

Boolean array indicating which cards are involved in the current stack:
- `stack[card_index] = True` if that card is part of the active stack
- Tracks ongoing card interactions/resolutions

**Total: 52 booleans**

### Effect-Shown (52 booleans)

Boolean array indicating which cards are shown by game effects:
- `effect_shown[card_index] = True` if that card is revealed by an effect
- Tracks public information from card abilities

**Total: 52 booleans**

### Deck

The deck is also represented as a boolean array (part of zones):
- `deck[card_index] = True` if the card is still in the deck

## Card Indexing

Cards are indexed using the formula:
```
card_index = 13 * suit + rank
```

Where:
- `suit ∈ [0, 3]` (Clubs, Diamonds, Hearts, Spades)
- `rank ∈ [0, 12]` (Ace through King)

## Total Input Dimension

**468 boolean features** concatenated into a single input vector:
- Zones: 364 booleans
- Stack: 52 booleans
- Effect-shown: 52 booleans
- **Total: 468 dimensions**

## Network Input Processing

The [[Neural Network Architecture]]:
1. Receives observation dictionary
2. Extracts all boolean arrays (zones, stack, effect_shown)
3. Concatenates them into a single 468-dimensional vector
4. Passes through fully connected layers

See [[Neural Network Basics]] for how networks process inputs.

## Design Rationale

### Why Boolean Arrays?

1. **Simplicity**: Direct presence/absence indication
2. **No Embeddings**: Avoids learning card representations
3. **Uniformity**: All inputs have the same format (52-length booleans)
4. **Interpretability**: Clear binary features

### Advantages

- No need for [[Neural Network Architecture|embedding layers]]
- Lower parameter count
- Explicit feature representation
- Easy to visualize and debug

### Trade-offs

- Fixed-size representation (always 52 cards)
- Sparse representation (many zeros)
- No explicit encoding of card relationships

## Related Concepts

- [[CuttleEnvironment]] - Environment that produces these observations
- [[Neural Network Architecture]] - Network that processes observations
- [[Action System]] - Actions taken based on observations
- [[Feature Engineering]] - Designing effective input representations

---
*Observation format for Cuttle DQN project*
