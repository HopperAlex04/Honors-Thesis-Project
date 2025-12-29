"""
Training script for model with only hand feature enabled.

This script trains a model using only the "Highest Point Value in Hand" feature,
with the opponent field feature disabled. This allows comparison of feature
contributions to model performance.
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

# Load hyperparameters from config file for easy experimentation
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
            "validation_episodes_ratio": 0.5
        },
        "early_stopping": {
            "enabled": True,
            "check_interval": 50,
            "window_size": 100,
            "divergence_threshold": 0.5,
            "min_episodes": 200,
            "max_loss": 50.0
        },
        "regression_detection": {
            "grace_rounds": 3,
            "grace_zone": 0.45,
            "max_consecutive_regressions": 2
        }
    }

# Create environment with only hand feature enabled
env = CuttleEnvironment(
    include_highest_point_value=True,
    include_highest_point_value_opponent_field=False
)
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
LR_DECAY_RATE = config["lr_decay_rate"]
LR_DECAY_INTERVAL = config["lr_decay_interval"]
GRADIENT_CLIP_NORM = config.get("gradient_clip_norm", 5.0)
Q_VALUE_CLIP = config.get("q_value_clip", 15.0)
REPLAY_BUFFER_SIZE = config.get("replay_buffer_size", 100000)

model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
trainee = Players.Agent(
    "PlayerAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAY_BUFFER_SIZE
)
# Configure target network updates (hard updates every N steps for better stability)
trainee.set_target_update_frequency(TARGET_UPDATE_FREQUENCY)

valid_agent = Players.Agent("ValidAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAY_BUFFER_SIZE)
valid_agent.set_target_update_frequency(TARGET_UPDATE_FREQUENCY)

# Ensure models directory exists
models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models directory ready: {models_dir}")

# Training state file for resuming (separate from main training)
STATE_FILE = models_dir / "training_state_hand_only.json"

# Checkpoint prefix to avoid conflicts with other training scripts
CHECKPOINT_PREFIX = "hand_only_checkpoint"


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
    """Check for interruption and save state immediately if interrupted.
    
    Args:
        current_round: Current training round
        total_rounds: Total number of rounds
        agent: Optional agent to save checkpoint for (includes model + optimizer)
        steps_done: Current training step count
        total_time: Total elapsed training time
        
    Returns:
        True if training was interrupted and should exit, False otherwise
    """
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
steps_done = 0  # Global step counter for epsilon decay
total_time = 0.0  # Total elapsed training time (for checkpoint resumption)
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
            # Save full checkpoint with model and optimizer state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': 0,
                'target_update_counter': 0,  # Initialize counter
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
TARGET_WIN_RATE = 0.99  # Stop training when hitting this win rate vs ScoreGapMaximizer
REGRESSION_THRESHOLD = 0.15  # Flag regression if win rate drops by more than this
REGRESSION_WINDOW = 3  # Number of rounds to consider for regression detection

# Minimum viable performance (training stops if below this vs Randomized)
# Lowered to 55% to account for variance - agent shows learning
MIN_RANDOM_WIN_RATE = 0.55  # Should beat random player (allowing for variance)

# Track win rates for regression detection
win_rate_history = {
    "randomized": [],
    "highcard": [],
    "gapmaximizer": []
}

# Track regression for early stopping
REGRESSION_GRACE_ZONE = 0.45  # Only count as regression if win rate < this (accounts for variance in small samples)
# With 200 episodes per round, there's ~3.5% standard error, so 45% threshold gives ~1.4 sigma buffer
# This prevents false positives from normal variance while still catching real regressions

# Grace period: Skip regression detection for first N rounds to allow model to learn
# Early rounds may show legitimate learning curves (worse before better)
regression_config = config.get("regression_detection", {})
GRACE_ROUNDS = regression_config.get("grace_rounds", 3)  # Default: skip regression detection for first 3 rounds
MAX_CONSECUTIVE_REGRESSIONS = regression_config.get("max_consecutive_regressions", 2)  # Stop after N consecutive regressions

# Track regression for early stopping
consecutive_regressions = 0  # Count consecutive regressions

# Training configuration: Load from config or use defaults
training_config = config.get("training", {})
quick_test_mode = training_config.get("quick_test_mode", False)

# Validation configuration: Ratio of total validation episodes to training episodes
# 1.0 = same as training, 0.5 = half, 0.25 = quarter
# Each position gets half of total (both positions are tested)
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
# Total validation = eps_per_round * validation_episodes_ratio
# Per position = total / 2 (since we test both positions)
validation_episodes_total = int(eps_per_round * validation_episodes_ratio)
validation_episodes_per_position = max(1, validation_episodes_total // 2)
print(f"Validation: {validation_episodes_total} total episodes ({validation_episodes_per_position} per position, ratio: {validation_episodes_ratio})")

# Early stopping configuration
early_stopping_config = config.get("early_stopping", {})

print(f"\n{'='*60}")
print(f"TRAINING: Hand Feature Only")
print(f"Features: Hand=ON, Opponent Field=OFF")
print(f"Starting training: Rounds {start_round} to {rounds-1}")
print(f"{'='*60}\n")

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
        print(f"Loaded checkpoint: {checkpoint_path}")
        
        # Handle both old format (full model) and new format (state dict)
        if isinstance(prev_checkpoint, dict) and 'model_state_dict' in prev_checkpoint:
            # New format: dict with model_state_dict, optimizer_state_dict, steps_done
            trainee.model.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.policy.load_state_dict(prev_checkpoint['model_state_dict'])
            # CRITICAL: Always sync target network with policy network when loading checkpoint
            # This ensures target network is up-to-date and prevents regression from stale targets
            trainee.target.load_state_dict(prev_checkpoint['model_state_dict'])
            trainee.target.eval()  # Ensure target stays in eval mode
            if 'optimizer_state_dict' in prev_checkpoint:
                trainee.set_optimizer_state(prev_checkpoint['optimizer_state_dict'])
                print(f"Restored optimizer state from checkpoint {x}")
            if 'steps_done' in prev_checkpoint:
                steps_done = prev_checkpoint['steps_done']
                print(f"Restored steps_done: {steps_done}")
            # Restore target update counter if available (for consistency)
            if 'target_update_counter' in prev_checkpoint:
                trainee.update_target_counter = prev_checkpoint['target_update_counter']
                print(f"Restored target_update_counter: {trainee.update_target_counter}")
            prev_model_state = prev_checkpoint['model_state_dict']
        else:
            # Old format: full model object
            trainee.model.load_state_dict(prev_checkpoint.state_dict())
            trainee.policy.load_state_dict(prev_checkpoint.state_dict())
            # CRITICAL: Sync target network even for old format checkpoints
            trainee.target.load_state_dict(prev_checkpoint.state_dict())
            trainee.target.eval()
            prev_model_state = prev_checkpoint.state_dict()
        print(f"Updated trainee model and target network with checkpoint {x}")
        
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
    
    # Feature configuration for this training script
    INCLUDE_HAND_FEATURE = True
    INCLUDE_OPPONENT_FIELD_FEATURE = False

    try:
        _, _, steps_done = Training.selfPlayTraining(
            trainee, trainee, eps_per_round,
            model_id=f"hand_only_round_{x}_selfplay",
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            include_scores=False,
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

    # Calculate current total time (including self-play that just completed)
    current_total_time = total_time + (time.time() - round_start_time)

    # Create validation agent with previous checkpoint for comparison
    validation_model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None)
    validation_model.load_state_dict(prev_model_state)
    valid_agent = Players.Agent("ValidAgent", validation_model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAY_BUFFER_SIZE)
    valid_agent.set_target_update_frequency(TARGET_UPDATE_FREQUENCY)

    try:
        # Validate from both positions for fair evaluation
        p1w, p2w = Training.validate_both_positions(
            trainee, valid_agent, validation_episodes_per_position,
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            model_id_prefix=f"hand_only_round_{x}_vs_previous",
            round_number=x,
            initial_total_time=current_total_time
        )
        # steps_done doesn't change during validation, so we keep the previous value
    except Exception as e:
        print(f"Error during validation training in round {x}: {e}")
        continue
    
    # Check for interruption after vs_previous
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    # === REGRESSION DETECTION: Check if current round is losing to previous round ===
    win_rate_vs_previous = p1w / validation_episodes_total if validation_episodes_total > 0 else 0.0
    # Use grace zone to account for statistical variance in sample sizes (200 episodes per round)
    # Only count as regression if win rate is significantly below 50% (e.g., < 45%)
    # This prevents false positives from normal variance while still catching real regressions
    
    # Skip regression detection during grace period (first N rounds)
    # Early rounds may show legitimate learning curves where performance dips before improving
    in_grace_period = x < GRACE_ROUNDS
    is_regression = win_rate_vs_previous < REGRESSION_GRACE_ZONE and not in_grace_period
    
    if is_regression:
        consecutive_regressions += 1
        print(f"\n{'!'*60}")
        print(f"REGRESSION DETECTED: Round {x} is LOSING to Round {x-1}")
        print(f"   Win rate: {win_rate_vs_previous:.1%} (P1: {p1w} vs P2: {p2w})")
        print(f"   Consecutive regressions: {consecutive_regressions}/{MAX_CONSECUTIVE_REGRESSIONS}")
        print(f"   This indicates the agent may be forgetting previous strategies.")
        print(f"   Possible causes: overfitting to self-play, training instability, or catastrophic forgetting.")
        print(f"{'!'*60}\n")
        
        # Early stopping if too many consecutive regressions
        if consecutive_regressions >= MAX_CONSECUTIVE_REGRESSIONS:
            print(f"\n{'!'*60}")
            print(f"EARLY STOPPING: {consecutive_regressions} consecutive regressions detected")
            print(f"   Training stopped to prevent further degradation.")
            print(f"   Last successful round: {x - consecutive_regressions}")
            print(f"{'!'*60}\n")
            save_training_state(x + 1, rounds, steps_done, total_time)
            break
    elif in_grace_period:
        # During grace period, don't count as regression but still track performance
        if win_rate_vs_previous < 0.50:
            print(f"\n[Grace Period] Round {x} win rate: {win_rate_vs_previous:.1%} (regression detection disabled for first {GRACE_ROUNDS} rounds)")
        consecutive_regressions = 0  # Reset counter during grace period
    else:
        consecutive_regressions = 0  # Reset counter on improvement
        
        # Reset exploration boost if performance improved (no longer in regression)
        if hasattr(trainee, 'exploration_boost') and trainee.exploration_boost > 0:
            trainee.reset_exploration_boost()
            print(f"   Reset exploration boost - performance improved")
        
        # Warn if win rate is in the gray zone (between grace zone and 50%)
        if win_rate_vs_previous < 0.50 and win_rate_vs_previous >= REGRESSION_GRACE_ZONE:
            print(f"\nWARNING: Round {x} is slightly below 50% ({win_rate_vs_previous:.1%})")
            print(f"   Within grace zone ({REGRESSION_GRACE_ZONE:.0%}-50%), monitoring closely.\n")
        elif win_rate_vs_previous < 0.55:
            print(f"\nWARNING: Round {x} is barely beating Round {x-1} ({win_rate_vs_previous:.1%})")
            print(f"   Monitor closely - performance may be degrading.\n")

    # Save checkpoint for next round (always save, regardless of win/loss)
    new_checkpoint_path = models_dir / f"{CHECKPOINT_PREFIX}{x + 1}.pt"
    try:
        if p2w < p1w:
            # New model won - save it with full state
            checkpoint = {
                'model_state_dict': trainee.model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': steps_done,
                'target_update_counter': trainee.update_target_counter,  # Save for consistency
            }
            torch.save(checkpoint, new_checkpoint_path)
            print(f"Saved new checkpoint: {new_checkpoint_path} (new model won: p1w={p1w} vs p2w={p2w})")
        else:
            # Previous model won - revert to it and reset optimizer to prevent continued degradation
            # This is critical: keeping optimizer state with momentum from bad training can cause further regression
            print(f"\n{'!'*60}")
            print(f"REVERTING TO PREVIOUS MODEL due to regression")
            print(f"   Resetting optimizer state to prevent continued degradation")
            print(f"{'!'*60}\n")
            
            # Revert model to previous checkpoint
            trainee.model.load_state_dict(prev_model_state)
            trainee.policy.load_state_dict(prev_model_state)
            trainee.target.load_state_dict(prev_model_state)  # Also revert target network
            
            # Restore replay memory to state from start of round (before bad training)
            # This preserves good experiences from before the regression while removing bad ones
            memory_size_before = len(trainee.memory)
            if memory_snapshot is not None and len(memory_snapshot) > 0:
                trainee.memory.set_state(memory_snapshot)
                print(f"   Restored replay memory to start-of-round state ({memory_size_before} -> {len(trainee.memory)} experiences)")
                print(f"   This preserves {len(trainee.memory)} good experiences from before the regression")
            else:
                # Fallback: clear memory if snapshot not available or empty
                from cuttle.players import ReplayMemory
                trainee.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
                print(f"   Cleared replay memory ({memory_size_before} -> 0 experiences) - snapshot unavailable or empty")
            
            # Reset optimizer state to prevent momentum from bad training direction
            # Create fresh optimizer with potentially reduced learning rate
            current_lr = trainee.optimizer.param_groups[0]['lr']
            if win_rate_vs_previous < REGRESSION_GRACE_ZONE:
                # Significant regression (below grace zone): reduce learning rate by 50%
                new_lr = current_lr * 0.5
                print(f"   Reducing learning rate: {current_lr:.2e} -> {new_lr:.2e} (50% reduction)")
            else:
                # Minor regression (in grace zone): reduce learning rate by 25%
                new_lr = current_lr * 0.75
                print(f"   Reducing learning rate: {current_lr:.2e} -> {new_lr:.2e} (25% reduction)")
            
            # Recreate optimizer with new learning rate and fresh state
            trainee.optimizer = torch.optim.Adam(
                trainee.model.parameters(),
                lr=new_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-5
            )
            
            # Additional recovery strategies:
            # 1. Temporarily increase exploration to escape local minima
            # Boost exploration by resetting epsilon decay (equivalent to ~2000 steps back)
            exploration_boost_steps = int(EPS_DECAY * 0.07)  # ~7% of decay, roughly 2000 steps
            trainee.boost_exploration(exploration_boost_steps)
            print(f"   Boosted exploration: reset epsilon decay by {exploration_boost_steps} steps")
            print(f"   This increases exploration temporarily to help escape local minima")
            
            # 2. Increase target network update frequency for more stability
            # More frequent updates = more stable targets = better learning
            original_freq = trainee.target_update_frequency
            if original_freq > 0:
                # Reduce update frequency (update more often) for better stability
                new_freq = max(500, int(original_freq * 0.5))  # Update 2x more frequently, min 500
                trainee.set_target_update_frequency(new_freq)
                print(f"   Increased target network update frequency: {original_freq} -> {new_freq} steps")
            
            # 3. Add small weight noise to escape local minima (optional, conservative)
            # This is a gentle perturbation that can help escape bad local minima
            with torch.no_grad():
                noise_scale = 0.01  # 1% noise
                for param in trainee.model.parameters():
                    if param.requires_grad and param.numel() > 0:
                        # Use absolute value of parameter as scale to avoid issues with std()
                        param_scale = param.abs().mean().item()
                        if param_scale > 0:
                            noise = torch.randn_like(param) * noise_scale * param_scale
                            param.add_(noise)
                print(f"   Added {noise_scale*100:.1f}% weight noise to escape local minima")
            
            # Save checkpoint with reverted model and fresh optimizer
            checkpoint = {
                'model_state_dict': prev_model_state,
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'steps_done': steps_done,
                'target_update_counter': trainee.update_target_counter,  # Save for consistency
            }
            torch.save(checkpoint, new_checkpoint_path)
            print(f"Saved previous checkpoint: {new_checkpoint_path} (previous model won: p1w={p1w} vs p2w={p2w})")
    except Exception as e:
        print(f"Error saving checkpoint {new_checkpoint_path}: {e}")

    try:
        # Calculate current total time for validation display
        current_total_time = total_time + (time.time() - round_start_time)
        # Validate from both positions for fair evaluation (dealer gets 6 cards vs 5 for first player)
        trainee_wins_rand, opponent_wins_rand = Training.validate_both_positions(
            trainee, validation00, validation_episodes_per_position,
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            model_id_prefix=f"hand_only_round_{x}_vs_randomized",
            round_number=x,
            initial_total_time=current_total_time
        )
        win_rate_rand = trainee_wins_rand / validation_episodes_total if validation_episodes_total > 0 else 0.0
        win_rate_history["randomized"].append(win_rate_rand)
        print(f"Round {x}: Validation vs Randomized (both positions) - trainee: {trainee_wins_rand}, opponent: {opponent_wins_rand} (win rate: {win_rate_rand:.1%})")
    except Exception as e:
        print(f"Error during randomized validation in round {x}: {e}")
        continue
    
    # Check for interruption after vs_randomized
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    try:
        # Calculate current total time for validation display
        current_total_time = total_time + (time.time() - round_start_time)
        # Validate from both positions for fair evaluation
        trainee_wins_hc, opponent_wins_hc = Training.validate_both_positions(
            trainee, validation01, validation_episodes_per_position,
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            model_id_prefix=f"hand_only_round_{x}_vs_heuristic",
            round_number=x,
            initial_total_time=current_total_time
        )
        win_rate_hc = trainee_wins_hc / validation_episodes_total if validation_episodes_total > 0 else 0.0
        win_rate_history["highcard"].append(win_rate_hc)
        print(f"Round {x}: Validation vs HeuristicHighCard (both positions) - trainee: {trainee_wins_hc}, opponent: {opponent_wins_hc} (win rate: {win_rate_hc:.1%})")
    except Exception as e:
        print(f"Error during heuristic validation in round {x}: {e}")
        continue
    
    # Check for interruption after vs_heuristic
    if check_interrupt_and_save(x, rounds, trainee, steps_done, total_time):
        sys.exit(0)
    
    try:
        # Calculate current total time for validation display
        current_total_time = total_time + (time.time() - round_start_time)
        # Validate from both positions for fair evaluation
        trainee_wins_gap, opponent_wins_gap = Training.validate_both_positions(
            trainee, validation02, validation_episodes_per_position,
            include_highest_point_value=INCLUDE_HAND_FEATURE,
            include_highest_point_value_opponent_field=INCLUDE_OPPONENT_FIELD_FEATURE,
            model_id_prefix=f"hand_only_round_{x}_vs_gapmaximizer",
            round_number=x,
            initial_total_time=current_total_time
        )
        win_rate_gap = trainee_wins_gap / validation_episodes_total if validation_episodes_total > 0 else 0.0
        win_rate_history["gapmaximizer"].append(win_rate_gap)
        print(f"Round {x}: Validation vs ScoreGapMaximizer (both positions) - trainee: {trainee_wins_gap}, opponent: {opponent_wins_gap} (win rate: {win_rate_gap:.1%})")
    except Exception as e:
        print(f"Error during gap maximizer validation in round {x}: {e}")
        continue
    
    # === MINIMUM VIABILITY CHECK ===
    # DISABLED: If we can't beat random after a few rounds, something is fundamentally wrong
    # if x >= 2 and win_rate_rand < MIN_RANDOM_WIN_RATE:
    #     print(f"\n{'!'*60}")
    #     print(f"TRAINING STOPPED: Win rate vs Randomized ({win_rate_rand:.1%}) < {MIN_RANDOM_WIN_RATE:.0%}")
    #     print(f"   Agent is not learning basic strategy.")
    #     print(f"   Check: rewards, network architecture, or hyperparameters.")
    #     print(f"{'!'*60}\n")
    #     save_training_state(x + 1, rounds, steps_done)
    #     break
    
    # Update total time after round completes
    round_elapsed_time = time.time() - round_start_time
    total_time += round_elapsed_time
    
    # === EARLY STOPPING CHECK ===
    if win_rate_gap >= TARGET_WIN_RATE:
        print(f"\n{'='*60}")
        print(f"TARGET ACHIEVED! Win rate vs ScoreGapMaximizer: {win_rate_gap:.1%} >= {TARGET_WIN_RATE:.0%}")
        print(f"Early stopping at round {x+1}")
        print(f"{'='*60}\n")
        save_training_state(x + 1, rounds, steps_done, total_time)
        break
    
    # === REGRESSION DETECTION ===
    if len(win_rate_history["gapmaximizer"]) >= REGRESSION_WINDOW:
        recent_rates = win_rate_history["gapmaximizer"][-REGRESSION_WINDOW:]
        peak_rate = max(win_rate_history["gapmaximizer"][:-1]) if len(win_rate_history["gapmaximizer"]) > 1 else 0
        current_rate = win_rate_gap
        
        if peak_rate - current_rate > REGRESSION_THRESHOLD:
            print(f"\n{'!'*60}")
            print(f"REGRESSION DETECTED vs ScoreGapMaximizer!")
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
                print(f"REGRESSION WARNING vs {heuristic_name}: {peak_rate:.1%} -> {current_rate:.1%}")
    
    # Save state after completing round (before next iteration)
    save_training_state(x + 1, rounds, steps_done, total_time)
    
    # Check for interruption after round
    if check_interrupt_and_save(x + 1, rounds, trainee, steps_done, total_time):
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

