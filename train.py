"""
Unified training script for DQN agent in Cuttle card game.

This script trains a DQN agent through self-play with validation against
baseline opponents. Training runs from start to finish without checkpointing -
if interrupted, the run must be restarted.

For running multiple experiments, use the experiment management system:
    python scripts/experiment_manager.py init --name "experiment_name"
    python scripts/run_full_experiment.py
"""

import os
import sys
import json
import time
from pathlib import Path

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
        "embedding_size": 16,
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
            "validation_opponent": "randomized"
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

# Create environment
env = CuttleEnvironment()
actions = env.actions

# Load hyperparameters from config
EMBEDDING_SIZE = config["embedding_size"]
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

print(f"\n{'='*60}")
print(f"Network Type: {network_type}")
print(f"{'='*60}")

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

# Validation opponents
validation_config = config.get("training", {}).get("validation_opponent", "randomized")
validation_opponents = []
if validation_config == "randomized" or validation_config == "both":
    validation_opponents.append(("Randomized", Players.Randomized("Randomized")))
if validation_config == "gapmaximizer" or validation_config == "both":
    validation_opponents.append(("GapMaximizer", Players.ScoreGapMaximizer("GapMaximizer")))

if not validation_opponents:
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
print(f"TRAINING: {rounds} rounds, {eps_per_round} episodes per round")
print(f"Total training episodes: {rounds * eps_per_round}")
print(f"{'='*60}\n")

# Track win rates and timing
win_rate_history = {name: [] for name, _ in validation_opponents}
steps_done = 0
total_time = 0.0
training_start_time = time.time()

# Save initial model
initial_model_path = models_dir / "model_initial.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'network_type': network_type,
    'config': config,
}, initial_model_path)
print(f"Saved initial model: {initial_model_path}")

for round_num in range(rounds):
    round_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Round {round_num + 1}/{rounds} (steps: {steps_done})")
    print(f"{'='*60}")
    
    # Proactive learning rate scheduling: decay LR every N rounds
    if round_num > 0 and round_num % LR_DECAY_INTERVAL == 0:
        current_lr = trainee.optimizer.param_groups[0]['lr']
        new_lr = current_lr * LR_DECAY_RATE
        for param_group in trainee.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"LR decay: {current_lr:.2e} -> {new_lr:.2e}")
    
    # Self-play training
    try:
        _, _, steps_done = Training.selfPlayTraining(
            trainee, trainee, eps_per_round,
            model_id=f"round_{round_num}_selfplay",
            initial_steps=steps_done,
            round_number=round_num,
            initial_total_time=total_time,
            early_stopping_config=early_stopping_config
        )
    except Exception as e:
        print(f"Error during self-play training in round {round_num}: {e}")
        raise
    
    # Validation against opponents
    current_total_time = total_time + (time.time() - round_start_time)
    for opponent_name, opponent in validation_opponents:
        try:
            trainee_wins, opponent_wins = Training.validate_both_positions(
                trainee, opponent, validation_episodes_per_position,
                model_id_prefix=f"round_{round_num}_vs_{opponent_name.lower()}",
                round_number=round_num,
                initial_total_time=current_total_time
            )
            win_rate = trainee_wins / validation_episodes_total if validation_episodes_total > 0 else 0.0
            win_rate_history[opponent_name].append(win_rate)
            print(f"Round {round_num}: Validation vs {opponent_name} - trainee: {trainee_wins}, opponent: {opponent_wins} (win rate: {win_rate:.1%})")
        except Exception as e:
            print(f"Error during validation vs {opponent_name} in round {round_num}: {e}")
            raise
    
    # Update total time
    round_elapsed_time = time.time() - round_start_time
    total_time += round_elapsed_time
    print(f"Round {round_num + 1} completed in {round_elapsed_time:.1f}s (total: {total_time:.1f}s)")

# Training completed - save final model
final_model_path = models_dir / "model_final.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainee.get_optimizer_state(),
    'network_type': network_type,
    'config': config,
    'steps_done': steps_done,
    'total_time': total_time,
    'win_rate_history': win_rate_history,
}, final_model_path)

total_elapsed = time.time() - training_start_time

print(f"\n{'='*60}")
print("TRAINING COMPLETED SUCCESSFULLY")
print(f"{'='*60}")
print(f"Network type: {network_type}")
print(f"Total rounds: {rounds}")
print(f"Total episodes: {rounds * eps_per_round}")
print(f"Total steps: {steps_done}")
print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
print(f"\nFinal model saved: {final_model_path}")

print("\nFinal Win Rate Summary:")
for opponent_name, history in win_rate_history.items():
    if history:
        print(f"  vs {opponent_name}: {history[-1]:.1%} (peak: {max(history):.1%})")

print()
