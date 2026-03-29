#!/usr/bin/env python3
"""
Create a Double DQN experiment under the same conditions as the peak normal DQN run:
  large_hidden_embedding_run_01 in experiment_20260202_101706_scale11_vs_large_hidden_10rounds
  (62.80% peak vs GapMaximizer).

Same: architecture (large_hidden + embedding), rounds (10), eps_per_round (250),
training hyperparams (lr, eps_decay, validation_opponent "both", etc.).
Seed: random each time you create the experiment (override with --seed).
Only change: use_double_dqn = True.

Usage:
    python create_double_dqn_peak_conditions.py
    python create_double_dqn_peak_conditions.py --name my_double_dqn_peak
    python create_double_dqn_peak_conditions.py --network scale_11
"""

import json
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project root
project_root = Path(__file__).parent
experiments_dir = project_root / "experiments"

# Peak normal DQN experiment (in ThesisExpStroage)
PEAK_SOURCE = project_root / "ThesisExpStroage" / "experiment_20260202_101706_scale11_vs_large_hidden_10rounds"

# Same as other DQN scripts: scale-11 game_based wide→narrow [52*11, 13*11, 15*11]
SCALE_11_HIDDEN_LAYERS = [572, 143, 165]
NETWORK_CHOICES = ("large_hidden", "scale_11")

# Align with hyperparams_config.json / PPO experiments (train.py)
VALIDATION_EPISODES_PER_POSITION = 125
EXTENDED_VALIDATION_EVERY_N_ROUNDS = 5
EXTENDED_VALIDATION_EPISODES_PER_POSITION = 1000


def get_git_commit() -> str:
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


def create_double_dqn_peak_conditions(
    name: str = "double_dqn_peak_conditions",
    seed: Optional[int] = None,
    network: str = "large_hidden",
) -> Path:
    """
    Create an experiment with one run: Double DQN, embedding,
    same base hyperparams as the peak normal DQN experiment (PEAK_SOURCE).

    network:
      - large_hidden: [512] (matches peak large_hidden run)
      - scale_11: game_based scale 11, [572, 143, 165]

    If seed is None, a random seed in [0, 2**31) is chosen.
    """
    if network not in NETWORK_CHOICES:
        raise ValueError(f"network must be one of {NETWORK_CHOICES}, got {network!r}")
    if not PEAK_SOURCE.exists():
        raise FileNotFoundError(
            f"Peak source experiment not found: {PEAK_SOURCE}. "
            "Ensure ThesisExpStroage/experiment_20260202_101706_scale11_vs_large_hidden_10rounds exists."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "runs").mkdir(exist_ok=True)

    # Base config from peak experiment
    base_src = PEAK_SOURCE / "base_hyperparams_config.json"
    base_dst = experiment_path / "base_hyperparams_config.json"
    with open(base_src) as f:
        config = json.load(f)
    config["use_double_dqn"] = True
    if "training" not in config:
        config["training"] = {}
    # Same validation schedule as PPO / root hyperparams (125 episodes per seat per opponent)
    config["training"]["validation_episodes_per_position"] = VALIDATION_EPISODES_PER_POSITION
    config["training"]["validation_opponent"] = "both"
    config["training"]["extended_validation_every_n_rounds"] = EXTENDED_VALIDATION_EVERY_N_ROUNDS
    config["training"]["extended_validation_episodes_per_position"] = (
        EXTENDED_VALIDATION_EPISODES_PER_POSITION
    )
    with open(base_dst, "w") as f:
        json.dump(config, f, indent=2)

    if seed is None:
        seed = random.randrange(2**31)
    rounds = 40

    if network == "scale_11":
        run_id = "scale_11_embedding_run_01"
        run_entry = {
            "run_id": run_id,
            "network_type": "game_based",
            "input_type": "embedding",
            "scale": 11,
            "hidden_layers": SCALE_11_HIDDEN_LAYERS,
            "rounds": rounds,
            "run_number": 1,
            "seed": seed,
            "status": "pending",
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "final_metrics": None,
            "error_message": None,
        }
        arch_label = "scale_11 game_based + embedding [572, 143, 165]"
    else:
        run_id = "large_hidden_embedding_run_01"
        run_entry = {
            "run_id": run_id,
            "network_type": "large_hidden",
            "input_type": "embedding",
            "scale": None,
            "hidden_layers": [512],
            "rounds": rounds,
            "run_number": 1,
            "seed": seed,
            "status": "pending",
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "final_metrics": None,
            "error_message": None,
        }
        arch_label = "large_hidden + embedding [512]"

    runs = {run_id: run_entry}

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": (
            "Double DQN under same base hyperparams as peak normal DQN experiment: "
            f"{arch_label}, random seed (or --seed), {rounds} rounds, 250 eps/round, "
            "validation_opponent both. Only change vs that setup: use_double_dqn=True."
        ),
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "experiment_type": "double_dqn_peak_conditions",
        "network": network,
        "run_order": [run_id],
        "rounds_per_run": rounds,
        "total_runs": 1,
        "seeds": [seed],
        "source_experiment": str(PEAK_SOURCE.name),
    }

    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))

    print("=" * 70)
    print(f"Created experiment: {experiment_name}")
    print("=" * 70)
    print(f"Location: {experiment_path}\n")
    print("Conditions (base hyperparams from peak experiment):")
    print(f"  Architecture: {arch_label}")
    print(f"  Seed: {seed}")
    print(f"  Rounds: {rounds}, 250 eps/round")
    print(f"  use_double_dqn: True (only change)")
    print(f"  Validation: both (randomized + GapMaximizer)")
    print(f"\nRun with: python experiment_manager.py run")
    print("=" * 70)

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Double DQN experiment under peak normal DQN conditions"
    )
    parser.add_argument(
        "--name", "-n",
        default="double_dqn_peak_conditions",
        help="Experiment name suffix (default: double_dqn_peak_conditions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed RNG seed for the run (default: random in [0, 2**31))",
    )
    parser.add_argument(
        "--network",
        choices=NETWORK_CHOICES,
        default="large_hidden",
        help=(
            "Policy network: large_hidden [512] (default, peak large_hidden match) or "
            "scale_11 game_based [572,143,165] with scale=11"
        ),
    )
    args = parser.parse_args()
    create_double_dqn_peak_conditions(
        args.name, seed=args.seed, network=args.network
    )
