"""
Unified training script for DQN agent in Cuttle card game.

This script trains a DQN agent through self-play with simplified checkpointing
and validation. All hint-features have been removed - training uses only raw
game state (zones, stack, effect_shown as boolean arrays).
"""

import os
import sys
import json
import signal
import time
from pathlib import Path
from typing import Optional

# Limit CPU threads to 4 cores (must be set before importing torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add src directory to Python path for package imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
torch.set_num_threads(4)

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork, EmbeddingBasedNetwork, MultiEncoderNetwork
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

# Load hyperparameters from config file
CONFIG_FILE = Path(__file__).parent / "hyperparams_config.json"
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    print(f"Loaded hyperparameters from {CONFIG_FILE}")
else:
    # Fallback to defaults if config file doesn't exist
    print(f"Warning: Config file not found at {CONFIG_FILE}, using defaults")
    config = {
        "embedding_size": 16,  # Kept for backward compatibility, no longer used
        "batch_size": 64,
        "gamma": 0.90,
        "eps_start": 0.90,
        "eps_end": 0.05,
        "eps_decay": 28510,
        "tau": 0.005,
        "target_update_frequency": 2000,
        "learning_rate": 8e-5,
        "lr_decay_rate": 0.9,
        "lr_decay_interval": 2,
        "gradient_clip_norm": 5.0,
        "q_value_clip": 15.0,
        "replay_buffer_size": 100000,
        "training": {
            "rounds": 10,
            "eps_per_round": 500,
            "quick_test_mode": False,
            "quick_test_rounds": 3,
            "quick_test_eps_per_round": 100,
            "validation_episodes_ratio": 0.5,
            "validation_opponent": "randomized"  # Options: "randomized", "gapmaximizer", "both"
        },
        "early_stopping": {
            "enabled": True,
            "check_interval": 50,
            "window_size": 100,
            "divergence_threshold": 0.5,
            "min_episodes": 200,
            "max_loss": 50.0
        }
    }

# Create environment (no feature flags needed)
env = CuttleEnvironment()
actions = env.actions

# Load hyperparameters from config
EMBEDDING_SIZE = config["embedding_size"]  # Kept for backward compatibility
BATCH_SIZE = config["batch_size"]
GAMMA = config["gamma"]
EPS_START = config["eps_start"]
EPS_END = config["eps_end"]
EPS_DECAY = config["eps_decay"]
TAU = config["tau"]
TARGET_UPDATE_FREQUENCY = config["target_update_frequency"]
LR = config["learning_rate"]
LR_DECAY_RATE = config.get("lr_decay_rate", 0.9)
LR_DECAY_INTERVAL = config.get("lr_decay_interval", 2)
GRADIENT_CLIP_NORM = config.get("gradient_clip_norm", 5.0)
Q_VALUE_CLIP = config.get("q_value_clip", 15.0)
REPLAY_BUFFER_SIZE = config.get("replay_buffer_size", 100000)

# Network type selection
network_type = config.get("network_type", "boolean")
embedding_dim = config.get("embedding_dim", 32)

if network_type == "embedding":
    model = EmbeddingBasedNetwork(env.observation_space, embedding_dim=embedding_dim, num_actions=actions)
elif network_type == "multi_encoder":
    model = MultiEncoderNetwork(env.observation_space, num_actions=actions)
else:
    # Default: boolean network (current NeuralNetwork)
    model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
trainee = Players.Agent(
    "PlayerAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAY_BUFFER_SIZE
)
trainee.set_target_update_frequency(TARGET_UPDATE_FREQUENCY)

# Ensure models directory exists
models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models directory ready: {models_dir}")

# Training state file for resuming
STATE_FILE = models_dir / "training_state.json"

# Checkpoint prefix
CHECKPOINT_PREFIX = "checkpoint"


def save_training_state(current_round: int, total_rounds: int, steps_done: int = 0, total_time: float = 0.0) -> None:
    """Save current training state to resume later."""
    state = {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "last_checkpoint": current_round,
        "steps_done": steps_done,
        "total_time": total_time,
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Training state saved: Round {current_round}/{total_rounds}, Steps: {steps_done}, Time: {total_time:.2f}s")
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


def check_interrupt_and_save(current_round: int, total_rounds: int, agent=None, steps_done: int = 0, total_time: float = 0.0) -> bool:
    """Check for interruption and save state immediately if interrupted."""
    if training_interrupted:
        print(f"\nTraining interrupted during round {current_round}")
        save_training_state(current_round, total_rounds, steps_done, total_time)
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
steps_done = 0
total_time = 0.0
if saved_state:
    response = input(f"Resume training from round {saved_state['current_round']}? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        start_round = saved_state['current_round']
        steps_done = saved_state.get('steps_done', 0)
        total_time = saved_state.get('total_time', 0.0)
        print(f"Resuming training from round {start_round}, steps {steps_done}, total time: {total_time:.2f}s")
    else:
        print("Starting fresh training")
        clear_training_state()

# Save initial checkpoint only if starting fresh
if start_round == 0:
    try:
        checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{0}.pt"
        if not checkpoint_path.exists():
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': 0,
                'target_update_counter': 0,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved initial checkpoint: {checkpoint_path}")
        else:
            print(f"Initial checkpoint already exists: {checkpoint_path}")
    except Exception as e:
        print(f"Error saving initial checkpoint: {e}")
        raise

# Validation opponents
validation_config = config.get("training", {}).get("validation_opponent", "randomized")
validation_opponents = []
if validation_config == "randomized" or validation_config == "both":
    validation_opponents.append(("Randomized", Players.Randomized("Randomized")))
if validation_config == "gapmaximizer" or validation_config == "both":
    validation_opponents.append(("GapMaximizer", Players.ScoreGapMaximizer("GapMaximizer")))

if not validation_opponents:
    # Default to randomized if invalid config
    validation_opponents = [("Randomized", Players.Randomized("Randomized"))]

# Early stopping configuration
early_stopping_config = config.get("early_stopping", {})

# Training configuration
training_config = config.get("training", {})
quick_test_mode = training_config.get("quick_test_mode", False)
validation_episodes_ratio = training_config.get("validation_episodes_ratio", 0.5)

if quick_test_mode:
    rounds = training_config.get("quick_test_rounds", 3)
    eps_per_round = training_config.get("quick_test_eps_per_round", 100)
    print(f"\n{'='*60}")
    print("QUICK TEST MODE ENABLED - Fast hyperparameter experimentation")
    print(f"Rounds: {rounds}, Episodes per round: {eps_per_round}")
    print(f"{'='*60}\n")
else:
    rounds = training_config.get("rounds", 10)
    eps_per_round = training_config.get("eps_per_round", 500)

# Calculate validation episodes per position
validation_episodes_total = int(eps_per_round * validation_episodes_ratio)
validation_episodes_per_position = max(1, validation_episodes_total // 2)
print(f"Validation: {validation_episodes_total} total episodes ({validation_episodes_per_position} per position, ratio: {validation_episodes_ratio})")
print(f"Validation opponents: {[name for name, _ in validation_opponents]}")

print(f"\n{'='*60}")
print(f"TRAINING: Raw Game State Only (No Hint-Features)")
print(f"Starting training: Rounds {start_round} to {rounds-1}")
print(f"{'='*60}\n")

# Track win rates for display
win_rate_history = {name: [] for name, _ in validation_opponents}

for x in range(start_round, rounds):
    # Check for interruption before starting round
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    round_start_time = time.time()
    
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
        
        if isinstance(prev_checkpoint, dict) and 'model_state_dict' in prev_checkpoint:
            trainee.model.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.policy.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.target.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.target.eval()
            if 'optimizer_state_dict' in prev_checkpoint:
                trainee.set_optimizer_state(prev_checkpoint['optimizer_state_dict'])
            if 'steps_done' in prev_checkpoint:
                steps_done = prev_checkpoint['steps_done']
            if 'target_update_counter' in prev_checkpoint:
                trainee.update_target_counter = prev_checkpoint['target_update_counter']
        else:
            # Old format: full model object
            trainee.model.load_state_dict(prev_checkpoint.state_dict())
            trainee.policy.load_state_dict(prev_checkpoint.state_dict())
            trainee.target.load_state_dict(prev_checkpoint.state_dict())
            trainee.target.eval()
        
        # Proactive learning rate scheduling: decay LR every N rounds
        if x > 0 and x % LR_DECAY_INTERVAL == 0:
            current_lr = trainee.optimizer.param_groups[0]['lr']
            new_lr = current_lr * LR_DECAY_RATE
            for param_group in trainee.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Proactive LR decay: {current_lr:.2e} -> {new_lr:.2e} (round {x} is multiple of {LR_DECAY_INTERVAL})")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print(f"Skipping round {x}.")
        continue
    
    # Self-play training
    try:
        _, _, steps_done = Training.selfPlayTraining(
            trainee, trainee, eps_per_round,
            model_id=f"round_{x}_selfplay",
            initial_steps=steps_done,
            round_number=x,
            initial_total_time=total_time,
            early_stopping_config=early_stopping_config
        )
    except Exception as e:
        print(f"Error during self-play training in round {x}: {e}")
        continue
    
    # Check for interruption after self-play
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    # Save checkpoint for next round (always save latest)
    new_checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{x + 1}.pt"
    try:
        checkpoint = {
            'model_state_dict': trainee.model.state_dict(),
            'optimizer_state_dict': trainee.get_optimizer_state(),
            'steps_done': steps_done,
            'target_update_counter': trainee.update_target_counter,
        }
        torch.save(checkpoint, new_checkpoint_path)
        print(f"Saved checkpoint: {new_checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint {new_checkpoint_path}: {e}")
    
    # Validation against opponents
    current_total_time = total_time + (time.time() - round_start_time)
    for opponent_name, opponent in validation_opponents:
        try:
            trainee_wins, opponent_wins = Training.validate_both_positions(
                trainee, opponent, validation_episodes_per_position,
                model_id_prefix=f"round_{x}_vs_{opponent_name.lower()}",
                round_number=x,
                initial_total_time=current_total_time
            )
            win_rate = trainee_wins / validation_episodes_total if validation_episodes_total > 0 else 0.0
            win_rate_history[opponent_name].append(win_rate)
            print(f"Round {x}: Validation vs {opponent_name} - trainee: {trainee_wins}, opponent: {opponent_wins} (win rate: {win_rate:.1%})")
        except Exception as e:
            print(f"Error during validation vs {opponent_name} in round {x}: {e}")
    
    # Check for interruption after validation
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    # Update total time after round completes
    round_elapsed_time = time.time() - round_start_time
    total_time += round_elapsed_time
    
    # Save state after completing round
    save_training_state(x + 1, rounds, steps_done, total_time)

# Training completed successfully
print(f"\n{'='*60}")
print("Training completed successfully!")
print(f"{'='*60}")

# Print final summary
print("\nFinal Win Rate Summary:")
for opponent_name, history in win_rate_history.items():
    if history:
        print(f"  vs {opponent_name}: {history[-1]:.1%} (peak: {max(history):.1%})")
print()

clear_training_state()
