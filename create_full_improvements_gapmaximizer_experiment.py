#!/usr/bin/env python3
"""
Create an experiment that uses all documented improvements for training vs GapMaximizer:
- Validation enabled (vs GapMaximizer), per-position stats
- Prioritized Experience Replay (PER)
- Double DQN
- Reward mode: normalized_score_diff
- Exploration boost on validation regression (5000 steps)
- Extended training: 15 rounds, slower epsilon decay (40k), eps_end 0.2
- Position alternates each round

Usage:
    python create_full_improvements_gapmaximizer_experiment.py
    python create_full_improvements_gapmaximizer_experiment.py --name full_improvements --rounds 10
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

# Scale 11 game_based (wide→narrow): [52*11, 13*11, 15*11]
SCALE_11_HIDDEN_LAYERS = [572, 143, 165]
ROUNDS_PER_RUN = 8
RUNS_PER_ARCHITECTURE = 2


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root
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


def _run_template(
    run_id: str,
    run_number: int,
    seed: int,
    network_type: str,
    input_type: str = "embedding",
    scale: Optional[int] = None,
    hidden_layers: Optional[List[int]] = None,
) -> dict:
    """Build a run entry for runs_status.json."""
    return {
        "run_id": run_id,
        "network_type": network_type,
        "input_type": input_type,
        "scale": scale,
        "hidden_layers": hidden_layers or ([512] if network_type == "large_hidden" else SCALE_11_HIDDEN_LAYERS),
        "rounds": ROUNDS_PER_RUN,
        "run_number": run_number,
        "seed": seed,
        "status": "pending",
        "created_at": None,
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "final_metrics": None,
        "error_message": None,
    }


def create_full_improvements_gapmaximizer_experiment(
    name: str = "full_improvements_gapmaximizer",
    description: str = "",
    runs_per_arch: int = RUNS_PER_ARCHITECTURE,
    rounds: int = ROUNDS_PER_RUN,
) -> Path:
    """
    Create an experiment with all improvements: PER, Double DQN, normalized_score_diff,
    validation enabled, exploration boost on regression, extended rounds and exploration.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    (experiment_path / "runs").mkdir(exist_ok=True)

    total_runs = 2 * runs_per_arch
    seeds = generate_seeds(total_runs)

    runs = {}
    run_num = 0

    for i in range(runs_per_arch):
        run_num += 1
        run_id = f"scale_11_vs_gapmax_run_{i + 1:02d}"
        runs[run_id] = _run_template(
            run_id=run_id,
            run_number=run_num,
            seed=seeds[2 * i],
            network_type="game_based",
            input_type="embedding",
            scale=11,
            hidden_layers=SCALE_11_HIDDEN_LAYERS,
        )
        runs[run_id]["rounds"] = rounds

    for i in range(runs_per_arch):
        run_num += 1
        run_id = f"large_hidden_vs_gapmax_run_{i + 1:02d}"
        runs[run_id] = _run_template(
            run_id=run_id,
            run_number=run_num,
            seed=seeds[2 * i + 1],
            network_type="large_hidden",
            input_type="embedding",
            scale=None,
            hidden_layers=[512],
        )
        runs[run_id]["rounds"] = rounds

    run_order: List[str] = []
    for i in range(runs_per_arch):
        run_order.append(f"scale_11_vs_gapmax_run_{i + 1:02d}")
        run_order.append(f"large_hidden_vs_gapmax_run_{i + 1:02d}")

    base_desc = (
        f"All improvements vs GapMaximizer: PER, Double DQN, normalized_score_diff, "
        f"validation on (500 per position, half as often), exploration boost on regression, {rounds} rounds, 500 eps/round. "
        f"Slower eps_decay (40k), eps_end 0.2. {runs_per_arch} seeds per arch."
    )

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or base_desc,
        "experiment_type": "full_improvements_gapmaximizer",
        "train_vs_gapmaximizer": True,
        "skip_validation": False,
        "trainee_first_only": False,
        "use_double_dqn": True,
        "use_prioritized_replay": True,
        "reward_mode": "normalized_score_diff",
        "exploration_boost_on_regression_steps": 5000,
        "rounds_per_run": rounds,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "run_order": run_order,
        "runs_per_architecture": runs_per_arch,
        "total_runs": total_runs,
        "seeds": seeds,
    }

    # Copy base config and apply all improvements
    base_config_src = project_root / "hyperparams_config.json"
    base_config_dst = experiment_path / "base_hyperparams_config.json"
    if base_config_src.exists():
        shutil.copy(base_config_src, base_config_dst)
        with open(base_config_dst) as f:
            config = json.load(f)

        config["use_double_dqn"] = True
        config["use_prioritized_replay"] = True
        config["eps_decay"] = 40000
        config["eps_end"] = 0.2

        if "training" not in config:
            config["training"] = {}
        config["training"]["train_vs_gapmaximizer"] = True
        config["training"]["skip_validation"] = False
        config["training"]["trainee_first_only"] = False
        config["training"]["reward_mode"] = "normalized_score_diff"
        config["training"]["exploration_boost_on_regression_steps"] = 5000
        config["training"]["validation_opponent"] = "gapmaximizer"
        config["training"]["rounds"] = rounds
        config["training"]["eps_per_round"] = 500

        with open(base_config_dst, "w") as f:
            json.dump(config, f, indent=2)

    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))

    print(f"{'='*70}")
    print(f"Created experiment: {experiment_name}")
    print(f"{'='*70}\n")
    print(f"Location: {experiment_path}\n")
    print("Improvements enabled:")
    print("  - Validation: on (vs GapMaximizer, per-position stats)")
    print("  - PER: on")
    print("  - Double DQN: on")
    print("  - Reward: normalized_score_diff")
    print("  - Exploration boost on regression: 5000 steps")
    print("  - eps_decay: 40000, eps_end: 0.2")
    print(f"  - Rounds: {rounds}")
    print(f"\nArchitectures: scale_11 (game_based), large_hidden")
    print(f"Runs per architecture: {runs_per_arch} (total {total_runs} runs)")
    print(f"Run order:")
    for idx, rid in enumerate(run_order, 1):
        info = runs[rid]
        arch = "game_based " + str(info["hidden_layers"]) if info["network_type"] == "game_based" else "large_hidden [512]"
        print(f"  {idx}. {rid} — {arch}, seed {info['seed']}, {info['rounds']} rounds")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"{'='*70}\n")

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create experiment with all improvements vs GapMaximizer"
    )
    parser.add_argument("--name", "-n", default="full_improvements_gapmaximizer",
                        help="Experiment name")
    parser.add_argument("--description", "-d", default="",
                        help="Experiment description")
    parser.add_argument("--runs-per-arch", "-r", type=int, default=RUNS_PER_ARCHITECTURE,
                        help=f"Runs per architecture (default: {RUNS_PER_ARCHITECTURE})")
    parser.add_argument("--rounds", type=int, default=ROUNDS_PER_RUN,
                        help=f"Rounds per run (default: {ROUNDS_PER_RUN})")

    args = parser.parse_args()
    create_full_improvements_gapmaximizer_experiment(
        args.name,
        args.description,
        runs_per_arch=args.runs_per_arch,
        rounds=args.rounds,
    )
