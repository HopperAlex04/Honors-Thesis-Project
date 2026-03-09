"""
Unified training script for DQN agent in Cuttle card game.

This script trains a DQN agent through self-play with validation against
baseline opponents. Round checkpointing is enabled: each round saves
model_round_N.pt, and the best round by validation (vs GapMaximizer if present)
is saved as model_best.pt with best_round.json for later use.

Usage:
    python train.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np

# Limit CPU threads to 4 cores (must be set before importing torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add src directory to Python path for package imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
torch.set_num_threads(4)


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This ensures that experiments are reproducible when using the same seed.
    Critical for scientific validity of RL experiments.
    
    Args:
        seed: Random seed to use for all generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on CUDA (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import (
    NeuralNetwork,
    EmbeddingBasedNetwork,
    MultiEncoderNetwork,
    EmbeddingBasedNeuralNetwork,
    ActorCritic,
    EmbeddingActorCritic,
)
from cuttle import training as Training


def build_dqn_model_from_config(cfg, observation_space, num_actions):
    """Build DQN model from config. Used for main model and loading historical checkpoints."""
    net_type = cfg.get("network_type", "game_based")
    use_embeddings = cfg.get("use_embeddings", True)
    embedding_dim = cfg.get("embedding_dim", 52)
    zone_encoded_dim = cfg.get("zone_encoded_dim", 52)
    emb_size = cfg.get("embedding_size", 16)
    use_position_indicator = cfg.get("use_position_indicator", True)

    if use_embeddings:
        if net_type == "linear":
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[], use_position_indicator=use_position_indicator
            )
        if net_type == "large_hidden":
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[512], use_position_indicator=use_position_indicator
            )
        if net_type == "game_based":
            scale = cfg.get("game_based_scale", 2)
            layers = cfg.get("game_based_hidden_layers")
            hidden = layers or [52 * scale, 13 * scale, 15 * scale]
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=hidden, use_position_indicator=use_position_indicator
            )
    else:
        if net_type == "linear":
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[])
        if net_type == "large_hidden":
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[512])
        if net_type == "game_based":
            scale = cfg.get("game_based_scale", 2)
            layers = cfg.get("game_based_hidden_layers")
            hidden = layers or [52 * scale, 13 * scale, 15 * scale]
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=hidden)
        if net_type == "embedding":
            return EmbeddingBasedNetwork(observation_space, embedding_dim=embedding_dim, num_actions=num_actions)
        if net_type == "multi_encoder":
            return MultiEncoderNetwork(observation_space, num_actions=num_actions)

    return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[52, 13, 15])


def load_historical_agent_from_checkpoint(checkpoint_path: Path, env, actions, name: str = "Historical"):
    """Load a DQN agent from checkpoint for use as opponent. Acts greedily (no exploration)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", config)
    net_type = ckpt.get("network_type", cfg.get("network_type", "game_based"))

    if net_type == "ppo":
        raise ValueError("Historical opponent pool only supports DQN; PPO not implemented")

    model = build_dqn_model_from_config(cfg, env.observation_space, actions)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hist = Players.Agent(
        name, model, BATCH_SIZE, GAMMA,
        eps_start=0.0, eps_end=0.0, eps_decay=1, tau=TAU, lr=LR,
        replay_buffer_size=1024,
        use_prioritized_replay=False,
        use_double_dqn=USE_DOUBLE_DQN,
        q_value_clip=Q_VALUE_CLIP,
    )
    hist.set_target_update_frequency(0)
    return hist


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
USE_PRIORITIZED_REPLAY = config.get("use_prioritized_replay", False)
PER_ALPHA = config.get("per_alpha", 0.6)
PER_BETA = config.get("per_beta", 0.4)
PER_BETA_END = config.get("per_beta_end", 1.0)
PER_EPSILON = config.get("per_epsilon", 1e-6)
USE_DOUBLE_DQN = config.get("use_double_dqn", False)
if USE_PRIORITIZED_REPLAY:
    print("Prioritized Experience Replay (PER) enabled (alpha=%.2f, beta=%.2f->%.2f)" % (PER_ALPHA, PER_BETA, PER_BETA_END))
if USE_DOUBLE_DQN:
    print("Double DQN enabled (policy selects next action, target evaluates)")

# Random seed for reproducibility
RANDOM_SEED = config.get("random_seed", None)
if RANDOM_SEED is not None:
    set_random_seeds(RANDOM_SEED)
    print(f"Random seed set: {RANDOM_SEED}")
else:
    print("Warning: No random seed specified. Results will not be reproducible.")

# Algorithm: "dqn" (default) or "ppo"
algorithm = config.get("algorithm", "dqn")
network_type = config.get("network_type", "game_based")
embedding_dim = config.get("embedding_dim", 32)

print(f"\n{'='*60}")
print(f"Algorithm: {algorithm.upper()}")
print(f"Network Type: {network_type}")
print(f"{'='*60}")

# Model and trainee creation: PPO vs DQN
if algorithm == "ppo":
    ppo_config = config.get("ppo", {})
    # Default [128, 128] — avoid aggressive compression (e.g. [128, 12, 8]) which hurts learning
    ppo_hidden = ppo_config.get("hidden_layers", [128, 128])
    use_embeddings_ppo = config.get("use_embeddings", True)
    if use_embeddings_ppo:
        embedding_dim = config.get("embedding_dim", 52)
        zone_encoded_dim = config.get("zone_encoded_dim", 52)
        use_position_indicator = ppo_config.get("use_position_indicator", False)
        model = EmbeddingActorCritic(
            env.observation_space,
            num_actions=actions,
            embedding_dim=embedding_dim,
            zone_encoded_dim=zone_encoded_dim,
            hidden_layers=ppo_hidden,
            use_position_indicator=use_position_indicator,
        )
        print("Using PPO (EmbeddingActorCritic) with hidden_layers:", ppo_hidden, "use_position_indicator:", use_position_indicator)
    else:
        model = ActorCritic(env.observation_space, num_actions=actions, hidden_layers=ppo_hidden)
        print("Using PPO (ActorCritic boolean) with hidden_layers:", ppo_hidden)
    trainee = Players.PPOAgent(
        "PlayerAgent",
        model,
        lr=ppo_config.get("learning_rate", 3e-4),
        gamma=ppo_config.get("gamma", 0.99),
        clip_eps=ppo_config.get("clip_eps", 0.2),
        ppo_epochs=ppo_config.get("ppo_epochs", 4),
        value_coef=ppo_config.get("value_coef", 0.5),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
    )
    network_type = "ppo"
else:
    # New architecture types (DQN)
    # Check if we should use embeddings for preprocessing (recommended based on previous results)
    use_embeddings = config.get("use_embeddings", True)  # Default to True for better performance
    embedding_dim = config.get("embedding_dim", 52)
    zone_encoded_dim = config.get("zone_encoded_dim", 52)

    if use_embeddings:
        # Use embeddings for preprocessing (proven to work well: 62% vs 18% win rate)
        # Then apply different hidden layer architectures for comparison
        if network_type == "linear":
            use_pos = config.get("use_position_indicator", True)
            model = EmbeddingBasedNeuralNetwork(
                env.observation_space,
                num_actions=actions,
                embedding_dim=embedding_dim,
                zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[],  # Linear: no hidden layers
                use_position_indicator=use_pos
            )
            print("Using embeddings with linear architecture (no hidden layers)")
        elif network_type == "large_hidden":
            use_pos = config.get("use_position_indicator", True)
            model = EmbeddingBasedNeuralNetwork(
                env.observation_space,
                num_actions=actions,
                embedding_dim=embedding_dim,
                zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[512],
                use_position_indicator=use_pos
            )
            print("Using embeddings with large hidden layer (512 neurons)" + ("" if use_pos else ", no position indicator (468-dim)"))
        elif network_type == "game_based":
            game_based_scale = config.get("game_based_scale", 2)
            game_based_hidden_layers = config.get("game_based_hidden_layers", None)
            hidden_layers = game_based_hidden_layers or [52 * game_based_scale, 13 * game_based_scale, 15 * game_based_scale]
            use_pos = config.get("use_position_indicator", True)
            model = EmbeddingBasedNeuralNetwork(
                env.observation_space, num_actions=actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=hidden_layers, use_position_indicator=use_pos
            )
            print(f"Using embeddings with game-based architecture: {hidden_layers} (scale={game_based_scale})" + ("" if use_pos else ", no position (468-dim)"))
    else:
        # Original boolean-based architectures (for comparison)
        if network_type == "linear":
            model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None, hidden_layers=[])
        elif network_type == "large_hidden":
            model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None, hidden_layers=[512])
        elif network_type == "game_based":
            game_based_scale = config.get("game_based_scale", 2)
            game_based_hidden_layers = config.get("game_based_hidden_layers", None)
            hidden_layers = game_based_hidden_layers or [52 * game_based_scale, 13 * game_based_scale, 15 * game_based_scale]
            print(f"Game-based hidden layers: {hidden_layers} (scale={game_based_scale})")
            model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None, hidden_layers=hidden_layers)
        elif network_type == "embedding":
            model = EmbeddingBasedNetwork(env.observation_space, embedding_dim=embedding_dim, num_actions=actions)
        elif network_type == "multi_encoder":
            model = MultiEncoderNetwork(env.observation_space, num_actions=actions)
        elif network_type == "boolean":
            model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None, hidden_layers=[52])
        else:
            model = NeuralNetwork(env.observation_space, EMBEDDING_SIZE, actions, None, hidden_layers=[52, 13, 15])

    q_value_clip = config.get("q_value_clip", 100.0)
    trainee = Players.Agent(
        "PlayerAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAY_BUFFER_SIZE,
        use_prioritized_replay=USE_PRIORITIZED_REPLAY,
        per_alpha=PER_ALPHA,
        per_beta=PER_BETA,
        per_beta_end=PER_BETA_END,
        per_epsilon=PER_EPSILON,
        use_double_dqn=USE_DOUBLE_DQN,
        q_value_clip=q_value_clip,
    )
    trainee.set_target_update_frequency(TARGET_UPDATE_FREQUENCY)

# PPO: batch trajectory every N episodes before optimize(); DQN uses None (optimize every episode)
optimize_every_n_episodes = (
    config.get("ppo", {}).get("optimize_every_n_episodes", 10) if algorithm == "ppo" else None
)
if optimize_every_n_episodes is not None:
    print(f"PPO: optimizing every {optimize_every_n_episodes} episodes (batched trajectory)")

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
validation_episodes_per_position = training_config.get("validation_episodes_per_position")
if validation_episodes_per_position is not None:
    validation_episodes_total = validation_episodes_per_position * 2
    print(f"Validation: {validation_episodes_total} total episodes ({validation_episodes_per_position} per position, from config)")
else:
    validation_episodes_total = int(eps_per_round * validation_episodes_ratio)
    validation_episodes_per_position = max(1, validation_episodes_total // 2)
    print(f"Validation: {validation_episodes_total} total episodes ({validation_episodes_per_position} per position, ratio: {validation_episodes_ratio})")

extended_validation_every_n = training_config.get("extended_validation_every_n_rounds", 5)
extended_validation_per_position = training_config.get("extended_validation_episodes_per_position", 500)
if extended_validation_every_n > 0:
    print(f"Extended validation: every {extended_validation_every_n} rounds, {extended_validation_per_position} per position")
print(f"Validation opponents: {[name for name, _ in validation_opponents]}")

reward_mode = training_config.get("reward_mode", "binary")
print(f"Reward mode: {reward_mode}")

# Train vs GapMaximizer instead of self-play (100% episodes vs GapMaximizer)
train_vs_gapmaximizer = training_config.get("train_vs_gapmaximizer", False)
trainee_first_only = training_config.get("trainee_first_only", False)
skip_validation = training_config.get("skip_validation", False)
exploration_boost_on_regression_steps = training_config.get("exploration_boost_on_regression_steps", 0)
# Anti-collapse: mix self-play with vs GapMaximizer and historical checkpoints
curriculum_vs_gapmaximizer_ratio = training_config.get("curriculum_vs_gapmaximizer_ratio", 0.0)
selfplay_historical_opponent_ratio = training_config.get("selfplay_historical_opponent_ratio", 0.0)
gapmaximizer_opponent = Players.ScoreGapMaximizer("GapMaximizer")
if train_vs_gapmaximizer:
    print(f"Training: vs GapMaximizer (trainee first only: {trainee_first_only})")
elif curriculum_vs_gapmaximizer_ratio > 0 or selfplay_historical_opponent_ratio > 0:
    parts = []
    if curriculum_vs_gapmaximizer_ratio > 0:
        parts.append(f"{curriculum_vs_gapmaximizer_ratio:.0%} vs GapMaximizer")
    if selfplay_historical_opponent_ratio > 0:
        parts.append(f"{selfplay_historical_opponent_ratio:.0%} vs historical")
    parts.append("rest self-play")
    print(f"Training: mixed ({', '.join(parts)})")
if skip_validation:
    print("Validation: skipped")
if exploration_boost_on_regression_steps > 0:
    print(f"Exploration boost on regression: {exploration_boost_on_regression_steps} steps")

print(f"\n{'='*60}")
print(f"TRAINING: {rounds} rounds, {eps_per_round} episodes per round")
print(f"Total training episodes: {rounds * eps_per_round}")
print(f"{'='*60}\n")

# Track win rates and timing
win_rate_history = {name: [] for name, _ in validation_opponents}
steps_done = 0
total_time = 0.0
training_start_time = time.time()

# Best-model tracking (by validation win rate; prefer GapMaximizer if present)
best_metric_opponent = None
for name, _ in validation_opponents:
    if "gapmaximizer" in name.lower():
        best_metric_opponent = name
        break
if best_metric_opponent is None and validation_opponents:
    best_metric_opponent = validation_opponents[0][0]
best_win_rate = -1.0
best_round = -1

# Save initial model
initial_model_path = models_dir / "model_initial.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'network_type': network_type,
    'random_seed': RANDOM_SEED,
    'config': config,
}, initial_model_path)
print(f"Saved initial model: {initial_model_path}")

for round_num in range(rounds):
    round_start_time = time.time()
    # Note: exploration_boost set after previous round's validation applies to THIS round's training.
    # We clear it at end of this round (when we don't boost again) so it only lasts one round.
    
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
    
    # Training: self-play or vs GapMaximizer
    try:
        if train_vs_gapmaximizer and gapmaximizer_opponent is not None:
            # trainee_first_only: trainee always P1. Else: alternate position each round.
            trainee_is_p1 = trainee_first_only or (round_num % 2 == 0)
            if trainee_is_p1:
                _, _, steps_done = Training.selfPlayTraining(
                    trainee, gapmaximizer_opponent, eps_per_round,
                    model_id=f"round_{round_num}_vs_gapmaximizer",
                    initial_steps=steps_done,
                    round_number=round_num,
                    initial_total_time=total_time,
                    early_stopping_config=early_stopping_config,
                    reward_mode=reward_mode,
                    optimize_every_n_episodes=optimize_every_n_episodes,
                )
            else:
                # Trainee as P2 this round
                _, _, steps_done = Training.selfPlayTraining(
                    gapmaximizer_opponent, trainee, eps_per_round,
                    model_id=f"round_{round_num}_vs_gapmaximizer",
                    initial_steps=steps_done,
                    round_number=round_num,
                    initial_total_time=total_time,
                    early_stopping_config=early_stopping_config,
                    reward_mode=reward_mode,
                    optimize_every_n_episodes=optimize_every_n_episodes,
                )
        else:
            # Self-play with optional mixed opponents to reduce collapse
            eps_gap = int(eps_per_round * curriculum_vs_gapmaximizer_ratio)
            eps_hist = int(eps_per_round * selfplay_historical_opponent_ratio) if round_num > 0 else 0
            eps_self = eps_per_round - eps_gap - eps_hist
            eps_self = max(0, eps_self)

            def run_block(p1, p2, n, mid):
                global steps_done
                if n <= 0:
                    return
                _, _, steps_done = Training.selfPlayTraining(
                    p1, p2, n,
                    model_id=mid,
                    initial_steps=steps_done,
                    round_number=round_num,
                    initial_total_time=total_time,
                    early_stopping_config=early_stopping_config,
                    reward_mode=reward_mode,
                    optimize_every_n_episodes=optimize_every_n_episodes,
                )

            # Block 1: vs GapMaximizer (position alternates within round)
            if eps_gap > 0:
                half = eps_gap // 2
                if half > 0:
                    run_block(trainee, gapmaximizer_opponent, half, f"round_{round_num}_vs_gapmax_curriculum_first")
                if eps_gap - half > 0:
                    run_block(gapmaximizer_opponent, trainee, eps_gap - half, f"round_{round_num}_vs_gapmax_curriculum_second")

            # Block 2: vs historical checkpoint (sample one per round)
            if eps_hist > 0:
                hist_round = random.randint(0, round_num - 1)
                ckpt = models_dir / f"model_round_{hist_round}.pt"
                if ckpt.exists():
                    hist_agent = load_historical_agent_from_checkpoint(ckpt, env, actions, f"Historical_r{hist_round}")
                    half = eps_hist // 2
                    if half > 0:
                        run_block(trainee, hist_agent, half, f"round_{round_num}_vs_historical_first")
                    if eps_hist - half > 0:
                        run_block(hist_agent, trainee, eps_hist - half, f"round_{round_num}_vs_historical_second")
                else:
                    run_block(trainee, trainee, eps_hist, f"round_{round_num}_selfplay")

            # Block 3: pure self-play
            if eps_self > 0:
                run_block(trainee, trainee, eps_self, f"round_{round_num}_selfplay")
    except Exception as e:
        print(f"Error during training in round {round_num}: {e}")
        raise

    # Validation against opponents (skipped if skip_validation)
    current_total_time = total_time + (time.time() - round_start_time)
    if not skip_validation:
        # Every N rounds use extended validation (500 per position) for robust checkpoint analysis
        use_extended = extended_validation_every_n > 0 and (round_num + 1) % extended_validation_every_n == 0
        eps_for_validation = extended_validation_per_position if use_extended else validation_episodes_per_position
        if use_extended:
            print(f"Extended validation this round: {eps_for_validation} per position")
        for opponent_name, opponent in validation_opponents:
            try:
                trainee_wins, opponent_wins, wins_as_p1, wins_as_p2 = Training.validate_both_positions(
                    trainee, opponent, eps_for_validation,
                    model_id_prefix=f"round_{round_num}_vs_{opponent_name.lower()}",
                    round_number=round_num,
                    initial_total_time=current_total_time
                )
                val_total = eps_for_validation * 2
                win_rate = trainee_wins / val_total if val_total > 0 else 0.0
                win_rate_history[opponent_name].append(win_rate)
                rate_p1 = wins_as_p1 / eps_for_validation if eps_for_validation > 0 else 0.0
                rate_p2 = wins_as_p2 / eps_for_validation if eps_for_validation > 0 else 0.0
                print(f"Round {round_num}: Validation vs {opponent_name} - trainee: {trainee_wins}, opponent: {opponent_wins} (win rate: {win_rate:.1%}) [as P1: {rate_p1:.1%}, as P2: {rate_p2:.1%}]")
            except Exception as e:
                print(f"Error during validation vs {opponent_name} in round {round_num}: {e}")
                raise

        # Exploration boost on regression: if win rate dropped for 2 consecutive rounds, boost epsilon for next round
        # (Requiring 2 consecutive regressions avoids death spiral from boosting on every small dip.)
        if (
            exploration_boost_on_regression_steps > 0
            and best_metric_opponent
            and hasattr(trainee, "boost_exploration")
            and hasattr(trainee, "reset_exploration_boost")
        ):
            hist = win_rate_history.get(best_metric_opponent, [])
            two_in_a_row = (
                len(hist) >= 3
                and hist[-1] < hist[-2]
                and hist[-2] < hist[-3]
            )
            if two_in_a_row:
                trainee.boost_exploration(exploration_boost_on_regression_steps)
                print(f"Validation regression vs {best_metric_opponent} (2 rounds): {hist[-3]:.1%} -> {hist[-2]:.1%} -> {hist[-1]:.1%}. Boosting exploration by {exploration_boost_on_regression_steps} steps for next round.")
            else:
                # Clear boost after one round of use so it doesn't persist
                trainee.reset_exploration_boost()

    # Update total time
    round_elapsed_time = time.time() - round_start_time
    total_time += round_elapsed_time
    print(f"Round {round_num + 1} completed in {round_elapsed_time:.1f}s (total: {total_time:.1f}s)")
    
    # Round checkpoint: save this round's model for later use
    round_checkpoint_path = models_dir / f"model_round_{round_num}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainee.get_optimizer_state(),
        'network_type': network_type,
        'random_seed': RANDOM_SEED,
        'config': config,
        'round': round_num,
        'steps_done': steps_done,
        'total_time': total_time,
        'win_rate_history': dict(win_rate_history),
    }, round_checkpoint_path)
    print(f"Saved round checkpoint: {round_checkpoint_path}")
    
    # Best-model tracking: update best if this round is best by validation (prefer GapMaximizer)
    if best_metric_opponent and win_rate_history.get(best_metric_opponent):
        current_win_rate = win_rate_history[best_metric_opponent][-1]
        if current_win_rate >= best_win_rate:
            best_win_rate = current_win_rate
            best_round = round_num
            best_model_path = models_dir / "model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainee.get_optimizer_state(),
                'network_type': network_type,
                'random_seed': RANDOM_SEED,
                'config': config,
                'round': round_num,
                'steps_done': steps_done,
                'total_time': total_time,
                'win_rate_history': dict(win_rate_history),
                'best_round': best_round,
                'best_win_rate': best_win_rate,
                'best_metric_opponent': best_metric_opponent,
            }, best_model_path)
            with open(models_dir / "best_round.json", "w") as f:
                json.dump({
                    "best_round": best_round,
                    "best_win_rate": best_win_rate,
                    "best_metric_opponent": best_metric_opponent,
                }, f, indent=2)
            print(f"New best model (round {round_num}, vs {best_metric_opponent}: {best_win_rate:.1%}): {best_model_path}")

# Training completed - save final model
final_model_path = models_dir / "model_final.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainee.get_optimizer_state(),
    'network_type': network_type,
    'random_seed': RANDOM_SEED,
    'config': config,
    'steps_done': steps_done,
    'total_time': total_time,
    'win_rate_history': win_rate_history,
    'final_win_rate': (win_rate_history.get(validation_opponents[0][0]) or [])[-1] if validation_opponents and (win_rate_history.get(validation_opponents[0][0]) or []) else None,
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
if skip_validation:
    print("  (Validation was skipped)")
else:
    for opponent_name, history in win_rate_history.items():
        if history:
            print(f"  vs {opponent_name}: {history[-1]:.1%} (peak: {max(history):.1%})")
if best_metric_opponent and best_round >= 0:
    print(f"\nBest checkpoint: round {best_round} (vs {best_metric_opponent}: {best_win_rate:.1%})")
    print(f"  Use models/model_best.pt or models/model_round_{best_round}.pt for best validation model.")

print()
