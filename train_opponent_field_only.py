"""
Training script for model with only opponent field feature enabled.

This script trains a model using only the "Highest Point Value in Opponent Field" feature,
with the hand feature disabled. This allows comparison of feature
contributions to model performance.
"""

import os
import sys
import json
import signal
from pathlib import Path
from typing import Optional

# Limit CPU threads to 4 cores (must be set before importing torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add src directory to Python path for package imports
# This allows imports to work without installing the package
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
torch.set_num_threads(4)

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

# Balanced hyperparameters for Cuttle training
EMBEDDING_SIZE = 16   # Good for 54 vocab size (cards + special values)
BATCH_SIZE = 64       # Balanced for CPU training
GAMMA = 0.92          # Medium-term focus (~25 steps significant)
EPS_START = 0.90      # Start with 90% exploration
EPS_END = 0.05        # Maintain 5% exploration when trained
EPS_DECAY = 12000     # Reach exploitation after ~2-3 rounds
TAU = 0.01            # Soft update rate (for future target network)
LR = 3e-4             # Moderate learning rate for faster initial learning

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


def save_training_state(current_round: int, total_rounds: int, steps_done: int = 0) -> None:
    """Save current training state to resume later."""
    state = {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "last_checkpoint": current_round,
        "steps_done": steps_done,
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Training state saved: Round {current_round}/{total_rounds}, Steps: {steps_done}")
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


def check_interrupt_and_save(current_round: int, total_rounds: int, agent=None, steps_done: int = 0) -> bool:
    """Check for interruption and save state immediately if interrupted.
    
    Args:
        current_round: Current training round
        total_rounds: Total number of rounds
        agent: Optional agent to save checkpoint for (includes model + optimizer)
        steps_done: Current training step count
        
    Returns:
        True if training was interrupted and should exit, False otherwise
    """
    if training_interrupted:
        print(f"\nTraining interrupted during round {current_round}")
        save_training_state(current_round, total_rounds, steps_done)
        if agent is not None:
            # Save full checkpoint with model, optimizer, and steps
            interrupt_checkpoint = models_dir / f"{CHECKPOINT_PREFIX}{current_round}_interrupted.pt"
            try:
                checkpoint = {
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.get_optimizer_state(),
                    'steps_done': steps_done,
                }
                torch.save(checkpoint, interrupt_checkpoint)
                print(f"Saved interrupt checkpoint: {interrupt_checkpoint}")
            except Exception as e:
                print(f"Warning: Could not save interrupt checkpoint: {e}")
        print("State saved. Run again to resume.")
        return True
    return False


# Check for existing training state to resume
saved_state = load_training_state()
start_round = 0
steps_done = 0  # Global step counter for epsilon decay
if saved_state:
    response = input(f"Resume training from round {saved_state['current_round']}? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        start_round = saved_state['current_round']
        steps_done = saved_state.get('steps_done', 0)
        print(f"Resuming training from round {start_round}, steps {steps_done}")
    else:
        print("Starting fresh training")
        clear_training_state()

# Save initial checkpoint only if starting fresh
if start_round == 0:
    try:
        checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{0}.pt"
        if not checkpoint_path.exists():
            # Save full checkpoint with model and optimizer state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': 0,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved initial checkpoint: {checkpoint_path}")
        else:
            print(f"Initial checkpoint already exists: {checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")
        raise

validation00 = Players.Randomized("Randomized")
validation01 = Players.HeuristicHighCard("HighCard")
validation02 = Players.ScoreGapMaximizer("GapMaximizer")

# Early stopping and regression detection settings
TARGET_WIN_RATE = 0.51  # Stop training when hitting this win rate vs ScoreGapMaximizer
REGRESSION_THRESHOLD = 0.15  # Flag regression if win rate drops by more than this
REGRESSION_WINDOW = 3  # Number of rounds to consider for regression detection

# Track win rates for regression detection
win_rate_history = {
    "randomized": [],
    "highcard": [],
    "gapmaximizer": []
}

rounds = 10
eps_per_round = 1000

print(f"\n{'='*60}")
print(f"TRAINING: Opponent Field Feature Only")
print(f"Features: Hand=OFF, Opponent Field=ON")
print(f"Starting training: Rounds {start_round} to {rounds-1}")
print(f"{'='*60}\n")

for x in range(start_round, rounds):
    # Check for interruption before starting round
    if check_interrupt_and_save(x, rounds, trainee, steps_done):
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"Round {x+1}/{rounds} (steps: {steps_done})")
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
        
        # Handle both old format (full model) and new format (state dict)
        if isinstance(prev_checkpoint, dict) and 'model_state_dict' in prev_checkpoint:
            trainee.model.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.policy.load_state_dict(prev_checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in prev_checkpoint:
                trainee.set_optimizer_state(prev_checkpoint['optimizer_state_dict'])
                print(f"Restored optimizer state from checkpoint {x}")
            if 'steps_done' in prev_checkpoint:
                steps_done = prev_checkpoint['steps_done']
                print(f"Restored steps_done: {steps_done}")
            prev_model_state = prev_checkpoint['model_state_dict']
        else:
            trainee.model.load_state_dict(prev_checkpoint.state_dict())
            trainee.policy.load_state_dict(prev_checkpoint.state_dict())
            prev_model_state = prev_checkpoint.state_dict()
        print(f"Updated trainee model with checkpoint {x}")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print(f"Skipping round {x}.")
        continue
    
    # Feature configuration for this training script
    INCLUDE_HAND_FEATURE = False
    INCLUDE_OPPONENT_FIELD_FEATURE = True

    try:
        _, _, steps_done = Training.selfPlayTraining(
            trainee, trainee, eps_per_round,
            model_id=f"opponent_field_only_round_{x}_selfplay",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            initial_steps=steps_done
        )
    except Exception as e:
        print(f"Error during self-play training in round {x}: {e}")
        continue
    
    # Check for interruption after self-play
    if check_interrupt_and_save(x, rounds, trainee, steps_done):
        sys.exit(0)

    # Create validation agent with previous checkpoint for comparison
    validation_model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
    validation_model.load_state_dict(prev_model_state)
    valid_agent = Players.Agent("ValidAgent", validation_model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

    try:
        p1w, p2w, steps_done = Training.selfPlayTraining(
            trainee, valid_agent, eps_per_round,
            validating=True,
            model_id=f"opponent_field_only_round_{x}_vs_previous",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            initial_steps=steps_done
        )
    except Exception as e:
        print(f"Error during validation training in round {x}: {e}")
        continue
    
    # Check for interruption after vs_previous
    if check_interrupt_and_save(x, rounds, trainee, steps_done):
        sys.exit(0)

    # Save checkpoint for next round (always save, regardless of win/loss)
    new_checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{x + 1}.pt"
    try:
        if p2w < p1w:
            checkpoint = {
                'model_state_dict': trainee.model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': steps_done,
            }
            torch.save(checkpoint, new_checkpoint_path)
            print(f"Saved new checkpoint: {new_checkpoint_path} (new model won: p1w={p1w} vs p2w={p2w})")
        else:
            checkpoint = {
                'model_state_dict': prev_model_state,
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': steps_done,
            }
            torch.save(checkpoint, new_checkpoint_path)
            trainee.model.load_state_dict(prev_model_state)
            trainee.policy.load_state_dict(prev_model_state)
            print(f"Saved previous checkpoint: {new_checkpoint_path} (previous model won: p1w={p1w} vs p2w={p2w})")
    except Exception as e:
        print(f"Error saving checkpoint {new_checkpoint_path}: {e}")

    try:
        p1w_rand, p2w_rand, steps_done = Training.selfPlayTraining(
            trainee, validation00, eps_per_round,
            validating=True,
            model_id=f"opponent_field_only_round_{x}_vs_randomized",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            initial_steps=steps_done
        )
        win_rate_rand = p1w_rand / eps_per_round
        win_rate_history["randomized"].append(win_rate_rand)
        print(f"Round {x}: Validation vs Randomized - p1w: {p1w_rand}, p2w: {p2w_rand} (win rate: {win_rate_rand:.1%})")
    except Exception as e:
        print(f"Error during randomized validation in round {x}: {e}")
        continue
    
    # Check for interruption after vs_randomized
    if check_interrupt_and_save(x, rounds, trainee, steps_done):
        sys.exit(0)
    
    try:
        p1w_hc, p2w_hc, steps_done = Training.selfPlayTraining(
            trainee, validation01, eps_per_round,
            validating=True,
            model_id=f"opponent_field_only_round_{x}_vs_heuristic",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            initial_steps=steps_done
        )
        win_rate_hc = p1w_hc / eps_per_round
        win_rate_history["highcard"].append(win_rate_hc)
        print(f"Round {x}: Validation vs HeuristicHighCard - p1w: {p1w_hc}, p2w: {p2w_hc} (win rate: {win_rate_hc:.1%})")
    except Exception as e:
        print(f"Error during heuristic validation in round {x}: {e}")
        continue
    
    # Check for interruption after vs_heuristic
    if check_interrupt_and_save(x, rounds, trainee, steps_done):
        sys.exit(0)
    
    try:
        p1w_gap, p2w_gap, steps_done = Training.selfPlayTraining(
            trainee, validation02, eps_per_round,
            validating=True,
            model_id=f"opponent_field_only_round_{x}_vs_gapmaximizer",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            initial_steps=steps_done
        )
        win_rate_gap = p1w_gap / eps_per_round
        win_rate_history["gapmaximizer"].append(win_rate_gap)
        print(f"Round {x}: Validation vs ScoreGapMaximizer - p1w: {p1w_gap}, p2w: {p2w_gap} (win rate: {win_rate_gap:.1%})")
    except Exception as e:
        print(f"Error during gap maximizer validation in round {x}: {e}")
        continue
    
    # === EARLY STOPPING CHECK ===
    if win_rate_gap >= TARGET_WIN_RATE:
        print(f"\n{'='*60}")
        print(f"🎯 TARGET ACHIEVED! Win rate vs ScoreGapMaximizer: {win_rate_gap:.1%} >= {TARGET_WIN_RATE:.0%}")
        print(f"Early stopping at round {x+1}")
        print(f"{'='*60}\n")
        save_training_state(x + 1, rounds, steps_done)
        break
    
    # === REGRESSION DETECTION ===
    if len(win_rate_history["gapmaximizer"]) >= REGRESSION_WINDOW:
        recent_rates = win_rate_history["gapmaximizer"][-REGRESSION_WINDOW:]
        peak_rate = max(win_rate_history["gapmaximizer"][:-1]) if len(win_rate_history["gapmaximizer"]) > 1 else 0
        current_rate = win_rate_gap
        
        if peak_rate - current_rate > REGRESSION_THRESHOLD:
            print(f"\n{'!'*60}")
            print(f"⚠️  REGRESSION DETECTED vs ScoreGapMaximizer!")
            print(f"    Peak win rate: {peak_rate:.1%}")
            print(f"    Current win rate: {current_rate:.1%}")
            print(f"    Drop: {(peak_rate - current_rate):.1%} (threshold: {REGRESSION_THRESHOLD:.0%})")
            print(f"    This may indicate self-play collapse.")
            print(f"{'!'*60}\n")
    
    # Check regression for other heuristics too
    for heuristic_name, history in [("Randomized", win_rate_history["randomized"]), 
                                      ("HighCard", win_rate_history["highcard"])]:
        if len(history) >= REGRESSION_WINDOW:
            peak_rate = max(history[:-1]) if len(history) > 1 else 0
            current_rate = history[-1]
            if peak_rate - current_rate > REGRESSION_THRESHOLD:
                print(f"⚠️  REGRESSION WARNING vs {heuristic_name}: {peak_rate:.1%} → {current_rate:.1%}")
    
    # Save state after completing round (before next iteration)
    save_training_state(x + 1, rounds, steps_done)
    
    # Check for interruption after round
    if check_interrupt_and_save(x + 1, rounds, trainee, steps_done):
        sys.exit(0)

# Training completed successfully
print(f"\n{'='*60}")
print("Training completed successfully!")
print(f"{'='*60}")

# Print final summary
print("\n📊 Final Win Rate Summary:")
if win_rate_history["randomized"]:
    print(f"  vs Randomized:        {win_rate_history['randomized'][-1]:.1%} (peak: {max(win_rate_history['randomized']):.1%})")
if win_rate_history["highcard"]:
    print(f"  vs HeuristicHighCard: {win_rate_history['highcard'][-1]:.1%} (peak: {max(win_rate_history['highcard']):.1%})")
if win_rate_history["gapmaximizer"]:
    print(f"  vs ScoreGapMaximizer: {win_rate_history['gapmaximizer'][-1]:.1%} (peak: {max(win_rate_history['gapmaximizer']):.1%})")
print()

clear_training_state()

