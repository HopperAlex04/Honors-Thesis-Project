# Cuttle Card Game - DQN Training

A reinforcement learning project for training DQN agents to play the Cuttle card game through self-play.

## Project Structure

```
.
├── src/
│   └── cuttle/              # Main package
│       ├── __init__.py      # Package exports
│       ├── environment.py   # Game environment (CuttleEnvironment)
│       ├── actions.py       # Action system (ActionRegistry, action classes)
│       ├── networks.py      # Neural network models (NeuralNetwork)
│       ├── players.py       # Player implementations (Agent, Randomized, etc.)
│       └── training.py      # Training utilities (selfPlayTraining)
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_environment.py  # Environment tests
│   ├── test_networks.py     # Neural network tests
│   └── test_players.py       # Player tests
├── scripts/                 # Scripts directory (optional)
├── models/                  # Saved model checkpoints
├── action_logs/             # Training logs
├── train_no_features.py     # Baseline training (no features)
├── train_hand_feature_only.py      # Hand feature only
├── train_opponent_field_only.py    # Opponent field feature only
├── train_both_features.py   # All features enabled (hand + opponent field + scores)
├── train_scores.py          # Scores feature only
├── setup.py                 # Package setup
├── requirements.txt         # Python dependencies
└── Dockerfile               # Docker configuration
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch
```

2. Install the package in editable mode:
```bash
pip install -e .
```

## Usage

Run training scripts for different feature configurations:
```bash
python train_no_features.py        # Baseline (no features)
python train_hand_feature_only.py  # Hand feature only
python train_opponent_field_only.py # Opponent field feature only
python train_both_features.py      # All features (hand + opponent field + scores)
python train_scores.py             # Scores feature only
```

Run tests:
```bash
python -m unittest discover tests
```

## Package Structure

The `cuttle` package provides:

- **`CuttleEnvironment`**: Gymnasium-compatible game environment
- **`ActionRegistry`**: Manages all possible game actions with integer indices
- **`NeuralNetwork`**: DQN neural network architecture
- **`Player`**: Base player interface
- **`Agent`**: DQN-based learning agent
- **`Randomized`**: Random action player
- **`HeuristicHighCard`**: Simple heuristic player
- **`ScoreGapMaximizer`**: Heuristic player that maximizes score gap
- **`selfPlayTraining`**: Self-play training function

## Versioning

This project uses Git tags for version management. To access a specific version:

```bash
# List all available versions
git tag -l

# Checkout a specific version (e.g., v1.0.0)
git checkout v1.0.0

# Return to the latest version
git checkout main

# View version information
git show v1.0.0
```

**Current Version:** v1.0.0

For detailed versioning instructions, see [VERSIONING.md](VERSIONING.md).

### Version History
- **v1.0.0** (2026-01-16) - Initial stable release with manual play functionality

## Development

The codebase follows Python best practices:
- Package structure with `src/` layout
- Snake_case naming for modules
- Proper `__init__.py` files
- Comprehensive test suite
- Type hints and docstrings

