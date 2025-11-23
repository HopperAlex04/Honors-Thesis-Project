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
├── Main.py                  # Main training script
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

Run the main training script:
```bash
python Main.py
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

## Development

The codebase follows Python best practices:
- Package structure with `src/` layout
- Snake_case naming for modules
- Proper `__init__.py` files
- Comprehensive test suite
- Type hints and docstrings

