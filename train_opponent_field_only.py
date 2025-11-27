"""
Training script for model with only opponent field feature enabled.

This script trains a model using only the "Highest Point Value in Opponent Field" feature,
with the hand feature disabled. This allows comparison of feature
contributions to model performance.
"""

import sys
import json
import signal
from pathlib import Path
from typing import Optional

# Add src directory to Python path for package imports
# This allows imports to work without installing the package
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork
from cuttle import training as Training

# Global flag for graceful shutdown
training_interrupted = False


def signal_handler(signum, frame):
    """Handle interrupt signal (Ctrl+C) gracefully."""
    global training_interrupted
    print("\n\nTraining interruption requested. Will finish current round and save state...")
    training_interrupted = True


# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create environment with only opponent field feature enabled
env = CuttleEnvironment(
    include_highest_point_value=False,
    include_highest_point_value_opponent_field=True
)
actions = env.actions

# Conservative hyperparameters (recommended starting point)
EMBEDDING_SIZE = 16  # Increased from 2 for better card representation
BATCH_SIZE = 128
GAMMA = 0.95  # Increased from 0.4 for better long-term planning
EPS_START = 0.95  # Slightly higher exploration
EPS_END = 0.05  # Keep some exploration even when trained
EPS_DECAY = 20000  # Slower decay for more exploration
TAU = 0.01  # Soft update rate
LR = 1e-4  # Slightly lower for stability

model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
trainee = Players.Agent(
    "PlayerAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR
)

valid_agent = Players.Agent("ValidAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

# Ensure models directory exists
models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models directory ready: {models_dir}")

# Training state file for resuming (separate from main training)
STATE_FILE = models_dir / "training_state_opponent_field_only.json"

# Checkpoint prefix to avoid conflicts with other training scripts
CHECKPOINT_PREFIX = "opponent_field_only_checkpoint"


def save_training_state(current_round: int, total_rounds: int) -> None:
    """Save current training state to resume later."""
    state = {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "last_checkpoint": current_round,
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Training state saved: Round {current_round}/{total_rounds}")
    except Exception as e:
        print(f"Warning: Could not save training state: {e}")


def load_training_state() -> Optional[dict]:
    """Load training state if it exists."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            print(f"Found existing training state: Round {state['current_round']}/{state['total_rounds']}")
            return state
        except Exception as e:
            print(f"Warning: Could not load training state: {e}")
            return None
    return None


def clear_training_state() -> None:
    """Clear training state file (called when training completes)."""
    if STATE_FILE.exists():
        try:
            STATE_FILE.unlink()
            print("Training state cleared (training completed)")
        except Exception as e:
            print(f"Warning: Could not clear training state: {e}")


# Check for existing training state to resume
saved_state = load_training_state()
start_round = 0
if saved_state:
    response = input(f"Resume training from round {saved_state['current_round']}? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        start_round = saved_state['current_round']
        print(f"Resuming training from round {start_round}")
    else:
        print("Starting fresh training")
        clear_training_state()

# Save initial checkpoint only if starting fresh
if start_round == 0:
    try:
        checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{0}.pt"
        if not checkpoint_path.exists():
            torch.save(model, checkpoint_path)
            print(f"Saved initial checkpoint: {checkpoint_path}")
        else:
            print(f"Initial checkpoint already exists: {checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")
        raise

validation00 = Players.Randomized("Randomized")
validation01 = Players.HeuristicHighCard("HighCard")
validation02 = Players.ScoreGapMaximizer("GapMaximizer")

rounds = 10
eps_per_round = 10

print(f"\n{'='*60}")
print(f"TRAINING: Opponent Field Feature Only")
print(f"Features: Hand=OFF, Opponent Field=ON")
print(f"Starting training: Rounds {start_round} to {rounds-1}")
print(f"{'='*60}\n")

for x in range(start_round, rounds):
    # Check for interruption before starting round
    if training_interrupted:
        print(f"\nTraining interrupted at round {x}")
        save_training_state(x, rounds)
        print("State saved. Run again to resume from this round.")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"Round {x+1}/{rounds}")
    print(f"{'='*60}")
    checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{x}.pt"
    
    # Check if checkpoint exists before loading
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found. Skipping round {x}.")
        continue
    
    # Load previous checkpoint and update trainee's model
    try:
        prev_checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        
        # Update trainee's model with the loaded checkpoint
        trainee.model.load_state_dict(prev_checkpoint.state_dict())
        trainee.policy.load_state_dict(prev_checkpoint.state_dict())
        print(f"Updated trainee model with checkpoint {x}")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print(f"Skipping round {x}.")
        continue
    
    try:
        Training.selfPlayTraining(trainee, trainee, eps_per_round, model_id=f"opponent_field_only_round_{x}_selfplay")
    except Exception as e:
        print(f"Error during self-play training in round {x}: {e}")
        continue

    # Create validation agent with previous checkpoint for comparison
    # Load state dict into a new model instance to avoid optimizer issues
    validation_model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
    validation_model.load_state_dict(prev_checkpoint.state_dict())
    valid_agent = Players.Agent("ValidAgent", validation_model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

    try:
        p1w, p2w = Training.selfPlayTraining(trainee, valid_agent, eps_per_round, True, model_id=f"opponent_field_only_round_{x}_vs_previous")
    except Exception as e:
        print(f"Error during validation training in round {x}: {e}")
        continue

    # Save checkpoint for next round (always save, regardless of win/loss)
    new_checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{x + 1}.pt"
    try:
        if p2w < p1w:
            # New model won - save it
            torch.save(trainee.model, new_checkpoint_path)
            print(f"Saved new checkpoint: {new_checkpoint_path} (new model won: p1w={p1w} vs p2w={p2w})")
        else:
            # Previous model won - save it instead
            torch.save(prev_checkpoint, new_checkpoint_path)
            # Also update trainee to use the previous model
            trainee.model.load_state_dict(prev_checkpoint.state_dict())
            trainee.policy.load_state_dict(prev_checkpoint.state_dict())
            print(f"Saved previous checkpoint: {new_checkpoint_path} (previous model won: p1w={p1w} vs p2w={p2w})")
    except Exception as e:
        print(f"Error saving checkpoint {new_checkpoint_path}: {e}")

    try:
        p1w, p2w = Training.selfPlayTraining(trainee, validation00, eps_per_round, True, model_id=f"opponent_field_only_round_{x}_vs_randomized")
        print(f"Round {x}: Validation vs Randomized - p1w: {p1w}, p2w: {p2w}")
    except Exception as e:
        print(f"Error during randomized validation in round {x}: {e}")
        continue
    
    try:
        p1w, p2w = Training.selfPlayTraining(trainee, validation01, eps_per_round, True, model_id=f"opponent_field_only_round_{x}_vs_heuristic")
        print(f"Round {x}: Validation vs HeuristicHighCard - p1w: {p1w}, p2w: {p2w}")
    except Exception as e:
        print(f"Error during heuristic validation in round {x}: {e}")
        continue
    
    try:
        p1w, p2w = Training.selfPlayTraining(trainee, validation02, eps_per_round, True, model_id=f"opponent_field_only_round_{x}_vs_gapmaximizer")
        print(f"Round {x}: Validation vs ScoreGapMaximizer - p1w: {p1w}, p2w: {p2w}")
    except Exception as e:
        print(f"Error during gap maximizer validation in round {x}: {e}")
        continue
    
    # Save state after completing round (before next iteration)
    save_training_state(x + 1, rounds)
    
    # Check for interruption after round
    if training_interrupted:
        print(f"\nTraining interrupted after round {x}")
        print("State saved. Run again to resume from next round.")
        sys.exit(0)

# Training completed successfully
print(f"\n{'='*60}")
print("Training completed successfully!")
print(f"{'='*60}\n")
clear_training_state()

