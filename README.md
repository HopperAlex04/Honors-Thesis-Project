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
├── scripts/                 # Utility scripts
│   ├── generate_metrics_graphs.py  # Generate training metrics visualizations
│   └── archive/             # Archived utility scripts
├── docs/                    # Documentation
│   └── archive/             # Archived analysis documents
├── models/                  # Saved model checkpoints
├── action_logs/             # Training logs
├── train.py                 # Unified training script
├── play_against_model.py    # Interactive play against trained model
├── hyperparams_config.json  # Hyperparameter configuration
├── setup.py                 # Package setup
├── requirements.txt         # Python dependencies
└── Dockerfile               # Docker configuration
```

## Installation

**Python:** 3.11+ (see `setup.py`).

### Option A: Virtual environment (recommended)

A virtual environment keeps project dependencies isolated. From the project root:

1. Create the venv (already done if you see a `.venv` folder):
   ```bash
   python3 -m venv .venv
   ```

2. Activate it:
   ```bash
   source .venv/bin/activate   # Linux / macOS
   # or on Windows:  .venv\Scripts\activate
   ```

3. Install dependencies and the package:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Run commands with the venv active, or call the venv Python directly:
   ```bash
   .venv/bin/python train.py
   .venv/bin/python scripts/plot_experiments.py --list
   ```

### Option B: System / user install

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

### Training

Run the unified training script:
```bash
python train.py
```

Training configuration is controlled via `hyperparams_config.json`:
- Training rounds and episodes per round
- Validation opponent selection (`validation_opponent`: "randomized", "gapmaximizer", or "both")
- Hyperparameters (learning rate, batch size, etc.)
- Early stopping configuration

The training script uses raw game state only (no hint-features) with uniform boolean array representation:
- All zones: boolean arrays of length 52
- Stack: boolean array of length 52
- Effect-Shown: boolean array of length 52

### Interactive Play

Play against a trained model:
```bash
python play_against_model.py
python play_against_model.py --checkpoint models/checkpoint0.pt
```

Run tests:
```bash
python -m unittest discover tests
```

## Package Structure

The `cuttle` package provides:

- **`CuttleEnvironment`**: Gymnasium-compatible game environment with uniform boolean array observations
- **`ActionRegistry`**: Manages all possible game actions with integer indices
- **`NeuralNetwork`**: DQN neural network architecture (no embeddings - all boolean arrays)
- **`Player`**: Base player interface
- **`Agent`**: DQN-based learning agent
- **`Randomized`**: Random action player
- **`HeuristicHighCard`**: Simple heuristic player
- **`ScoreGapMaximizer`**: Heuristic player that maximizes score gap
- **`selfPlayTraining`**: Self-play training function

## Observation Format

All observations use uniform boolean array representation:
- **Zones** (7 zones × 52 cards = 364 booleans): Hand, Field, Revealed, Off-Player Field, Off-Player Revealed, Deck, Scrap
- **Stack** (52 booleans): Which cards are involved in current stack
- **Effect-Shown** (52 booleans): Which cards are shown by effects
- **Total**: 468 boolean presence indicators concatenated into single input vector

No embeddings are used - all input is boolean presence arrays of length 52.

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

