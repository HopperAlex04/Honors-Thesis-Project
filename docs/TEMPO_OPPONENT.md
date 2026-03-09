# TempoPlayer: A Less Aggressive Opponent

**TempoPlayer** is a rule-based opponent that plays for score gap (like GapMaximizer) but with **tempo and risk** adjustments so it is less purely aggressive. It is not necessarily weaker—just a different style.

## How it differs from GapMaximizer

- **GapMaximizer**: Always picks the action that maximizes estimated score gap (my_score − opponent_score). Very aggressive: scores high, scuttles whenever it hurts the opponent more than us, Aces when opponent loses more.
- **TempoPlayer**: Uses the same gap estimate but then:
  1. **Scuttle**: Penalizes using *our* card. Prefers to scuttle with low-value cards and is less willing to burn a high card for a small gap gain.
  2. **Ace**: Penalizes *our* points on board. Less likely to board-wipe when we have a lot of points at risk.
  3. **Draw**: Gets a bonus so drawing is more competitive. When the best aggressive move is only slightly better than drawing, TempoPlayer can choose to draw instead (optional, controlled by `prefer_draw_when_close`).

So you get: less “always attack,” more drawing and selective use of Scuttle/Ace.

## Usage

```python
from cuttle.players import TempoPlayer

# Default: moderate penalties, prefer draw when close
opponent = TempoPlayer("Tempo")

# Tune behavior
opponent = TempoPlayer(
    "Tempo",
    scuttle_penalty=0.5,   # 0 = like GapMaximizer; higher = more reluctant to scuttle with high cards
    ace_penalty=0.5,      # 0 = like GapMaximizer; higher = more reluctant to Ace when we have points on board
    draw_bonus=0.4,       # added to Draw’s score so Draw is more attractive
    prefer_draw_when_close=True,
    draw_tolerance=0.5,   # prefer Draw when best_action is within this much of Draw’s score
)
```

Use `TempoPlayer` anywhere you would use `ScoreGapMaximizer` (e.g. as a validation opponent or training partner): same interface `getAction(observation, valid_actions, total_actions, steps_done, force_greedy)`.

## Using in training / validation

- **train.py**: Add a branch that builds a Tempo opponent instead of (or in addition to) GapMaximizer when you want a less aggressive validation/training partner.
- **Experiment scripts**: In the run config or `validation_opponents` list, add an entry that instantiates `TempoPlayer("Tempo")` instead of `ScoreGapMaximizer("GapMaximizer")` when you want to evaluate vs Tempo.

Example for validation:

```python
validation_opponents = [
    ("GapMaximizer", Players.ScoreGapMaximizer("GapMaximizer")),
    ("Tempo", Players.TempoPlayer("Tempo")),
]
```

## Design space for more opponents

- **More conservative**: Increase `scuttle_penalty` and `ace_penalty` (e.g. 0.8).
- **More draw-heavy**: Increase `draw_bonus` and `draw_tolerance`.
- **Closer to GapMaximizer**: Set `scuttle_penalty=0`, `ace_penalty=0`, `draw_bonus=0`, `prefer_draw_when_close=False`.

Other ideas you could add later: threshold-aware (prefer reaching 21 then defending), or a stochastic mix (e.g. 80% GapMaximizer / 20% random) for a different kind of “less aggressive” opponent.
