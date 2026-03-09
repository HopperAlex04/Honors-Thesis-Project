#!/usr/bin/env python3
"""
Create an experiment for PPO (Proximal Policy Optimization) training.

Supports reward shaping (binary vs normalized_score_diff), number of runs (seeds),
and backbone(s): both (scale_11 + large_hidden), scale_11, or large_hidden.
Training is vs GapMaximizer by default; validation can be enabled or skipped.

Usage:
    python create_ppo_experiment.py
    python create_ppo_experiment.py --backbones both --runs 2
    python create_ppo_experiment.py -b scale_11 -r 3 --ppo-hidden 512
    python create_ppo_experiment.py -R normalized_score_diff --backbones large_hidden
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Project root
project_root = Path(__file__).parent
experiments_dir = project_root / "experiments"

# Defaults
ROUNDS_PER_RUN = 40
RUNS_DEFAULT = 3
REWARD_MODES = ("binary", "normalized_score_diff")
BACKBONE_CHOICES = ("both", "scale_11", "large_hidden")
# Same as DQN experiments for fair comparison
SCALE_11_HIDDEN_LAYERS = [572, 143, 165]
LARGE_HIDDEN_LAYERS = [512]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def generate_seeds(count: int, base_seed: int = 42) -> List[int]:
    """Generate deterministic random seeds for reproducibility."""
    np.random.seed(base_seed)
    return [int(np.random.randint(1, 100000)) for _ in range(count)]


def create_ppo_experiment(
    name: str = "ppo",
    description: str = "",
    runs: int = RUNS_DEFAULT,
    reward_mode: str = "binary",
    rounds: int = ROUNDS_PER_RUN,
    train_vs_gapmaximizer: bool = True,
    skip_validation: bool = False,
    ppo_hidden_layers: Optional[List[int]] = None,
    backbones: str = "both",
) -> Path:
    """
    Create an experiment for PPO training with configurable reward mode, runs, and backbone(s).

    Args:
        name: Experiment display name (used in directory: experiment_<timestamp>_<name>).
        description: Optional longer description.
        runs: Number of runs per backbone when backbones='both'; total runs when a single backbone.
        reward_mode: "binary" (WIN/LOSS/DRAW) or "normalized_score_diff".
        rounds: Training rounds per run.
        train_vs_gapmaximizer: If True, train vs GapMaximizer; else self-play.
        skip_validation: If True, skip validation after each round.
        ppo_hidden_layers: Used only when backbones is 'scale_11' or 'large_hidden'; ignored when 'both'.
        backbones: "both" (scale_11 + large_hidden), "scale_11", or "large_hidden".

    Returns:
        Path to the created experiment directory.
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    if backbones not in BACKBONE_CHOICES:
        raise ValueError(f"backbones must be one of {BACKBONE_CHOICES}, got {backbones!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "runs").mkdir(exist_ok=True)

    if backbones == "both":
        # Two architectures, `runs` per architecture (same as full_improvements DQN)
        total_runs = 2 * runs
        seeds = generate_seeds(total_runs)
        run_order: List[str] = []
        runs_dict = {}
        run_num = 0
        for i in range(runs):
            run_num += 1
            run_id = f"ppo_scale11_run_{i + 1:02d}"
            run_order.append(run_id)
            runs_dict[run_id] = {
                "run_id": run_id,
                "network_type": "game_based",
                "input_type": "embedding",
                "run_number": run_num,
                "seed": seeds[2 * i],
                "rounds": rounds,
                "ppo_hidden_layers": SCALE_11_HIDDEN_LAYERS,
                "status": "pending",
                "created_at": None,
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "final_metrics": None,
                "error_message": None,
            }
        for i in range(runs):
            run_num += 1
            run_id = f"ppo_large_hidden_run_{i + 1:02d}"
            run_order.append(run_id)
            runs_dict[run_id] = {
                "run_id": run_id,
                "network_type": "large_hidden",
                "input_type": "embedding",
                "run_number": run_num,
                "seed": seeds[2 * i + 1],
                "rounds": rounds,
                "ppo_hidden_layers": LARGE_HIDDEN_LAYERS,
                "status": "pending",
                "created_at": None,
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "final_metrics": None,
                "error_message": None,
            }
        base_desc = (
            f"PPO training: scale_11 and large_hidden backbones, {runs} runs each ({total_runs} total). "
            f"Reward mode: {reward_mode}. "
            f"Train vs GapMaximizer: {train_vs_gapmaximizer}. Validation: {'skipped' if skip_validation else 'on'}."
        )
        metadata_ppo_layers = "scale_11 [572,143,165] and large_hidden [512]"
    else:
        # Single backbone: scale_11 or large_hidden
        hidden = SCALE_11_HIDDEN_LAYERS if backbones == "scale_11" else LARGE_HIDDEN_LAYERS
        if ppo_hidden_layers is not None:
            hidden = ppo_hidden_layers
        total_runs = runs
        seeds = generate_seeds(runs)
        run_order = []
        runs_dict = {}
        for i in range(runs):
            run_id = f"ppo_{backbones}_run_{i + 1:02d}"
            run_order.append(run_id)
            runs_dict[run_id] = {
                "run_id": run_id,
                "network_type": "game_based" if backbones == "scale_11" else "large_hidden",
                "input_type": "embedding",
                "run_number": i + 1,
                "seed": seeds[i],
                "rounds": rounds,
                "ppo_hidden_layers": hidden,
                "status": "pending",
                "created_at": None,
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "final_metrics": None,
                "error_message": None,
            }
        base_desc = (
            f"PPO training: {backbones} backbone, {runs} runs. "
            f"Reward mode: {reward_mode}. "
            f"Train vs GapMaximizer: {train_vs_gapmaximizer}. Validation: {'skipped' if skip_validation else 'on'}."
        )
        metadata_ppo_layers = str(hidden)

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or base_desc,
        "experiment_type": "ppo",
        "algorithm": "ppo",
        "reward_mode": reward_mode,
        "rounds_per_run": rounds,
        "train_vs_gapmaximizer": train_vs_gapmaximizer,
        "skip_validation": skip_validation,
        "backbones": backbones,
        "ppo_hidden_layers": metadata_ppo_layers,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "run_order": run_order,
        "total_runs": total_runs,
        "seeds": seeds,
    }

    # Base config: PPO + reward_mode + training options (per-run ppo.hidden_layers set by experiment_manager)
    base_config_src = project_root / "hyperparams_config.json"
    base_config_dst = experiment_path / "base_hyperparams_config.json"
    if base_config_src.exists():
        shutil.copy(base_config_src, base_config_dst)
        with open(base_config_dst) as f:
            config = json.load(f)
        config["algorithm"] = "ppo"
        if "ppo" not in config:
            config["ppo"] = {}
        # Default placeholder; each run overwrites via run_info["ppo_hidden_layers"]
        config["ppo"]["hidden_layers"] = config["ppo"].get("hidden_layers", [128, 128])
        if "training" not in config:
            config["training"] = {}
        config["training"]["reward_mode"] = reward_mode
        config["training"]["train_vs_gapmaximizer"] = train_vs_gapmaximizer
        config["training"]["skip_validation"] = skip_validation
        config["training"]["rounds"] = rounds
        config["training"]["validation_opponent"] = "gapmaximizer" if not skip_validation else "randomized"
        # PPO compensations: values in hyperparams_config.json are tuned for DQN and can hurt PPO
        if "early_stopping" not in config:
            config["early_stopping"] = {}
        config["early_stopping"]["enabled"] = False  # DQN-tuned max_loss/divergence thresholds inappropriate for PPO loss scale
        config["lr_decay_interval"] = 9999  # Disable DQN-style per-round LR decay; PPO uses fixed LR from ppo.learning_rate
        with open(base_config_dst, "w") as f:
            json.dump(config, f, indent=2)

    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs_dict, f, indent=2)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))

    print(f"{'='*70}")
    print(f"Created PPO experiment: {experiment_name}")
    print(f"{'='*70}\n")
    print(f"Location: {experiment_path}\n")
    print(f"  Algorithm: PPO")
    print(f"  Backbones: {backbones}")
    print(f"  Reward mode: {reward_mode}")
    print(f"  Train vs GapMaximizer: {train_vs_gapmaximizer}")
    print(f"  Validation: {'skipped' if skip_validation else 'on'}")
    print(f"  Rounds per run: {rounds}")
    print(f"  Total runs: {total_runs}")
    print(f"\nRun order:")
    for idx, rid in enumerate(run_order, 1):
        info = runs_dict[rid]
        layers = info.get("ppo_hidden_layers", [])
        print(f"  {idx}. {rid} — seed {info['seed']}, {info['rounds']} rounds, hidden {layers}")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"{'='*70}\n")
    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a PPO experiment with configurable reward shaping and number of runs"
    )
    parser.add_argument(
        "--name", "-n",
        default="ppo",
        help="Experiment name (default: ppo)",
    )
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Experiment description",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=RUNS_DEFAULT,
        help=f"Number of runs (seeds) (default: {RUNS_DEFAULT})",
    )
    parser.add_argument(
        "--reward-mode", "-R",
        default="binary",
        choices=REWARD_MODES,
        help="Reward shaping: binary (WIN/LOSS/DRAW) or normalized_score_diff (default: binary)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS_PER_RUN,
        help=f"Training rounds per run (default: {ROUNDS_PER_RUN})",
    )
    parser.add_argument(
        "--train-vs-gapmaximizer",
        action="store_true",
        default=True,
        help="Train vs GapMaximizer (default: True)",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Use self-play instead of vs GapMaximizer (sets train_vs_gapmaximizer=False)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation after each round",
    )
    parser.add_argument(
        "--ppo-hidden",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="PPO hidden layer sizes when using a single backbone (ignored if --backbones both). E.g. 128 128 or 512 (default: scale_11 [572,143,165] or large_hidden [512])",
    )
    parser.add_argument(
        "--backbones", "-b",
        default="both",
        choices=BACKBONE_CHOICES,
        help="Backbone(s): both (scale_11 + large_hidden), scale_11, or large_hidden (default: both)",
    )
    args = parser.parse_args()
    train_vs_gapmaximizer = not args.self_play
    create_ppo_experiment(
        name=args.name,
        description=args.description,
        runs=args.runs,
        reward_mode=args.reward_mode,
        rounds=args.rounds,
        train_vs_gapmaximizer=train_vs_gapmaximizer,
        skip_validation=args.skip_validation,
        ppo_hidden_layers=args.ppo_hidden,
        backbones=args.backbones,
    )
