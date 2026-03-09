#!/usr/bin/env python3
"""
Create an experiment where scale-11 and large_hidden models are trained vs GapMaximizer
with Double DQN enabled. Reward mode is configurable. Validation skipped; trainee
position alternates each round.

Usage:
    python create_double_dqn_gapmaximizer_experiment.py
    python create_double_dqn_gapmaximizer_experiment.py --name my_double_dqn
    python create_double_dqn_gapmaximizer_experiment.py --runs-per-arch 3
    python create_double_dqn_gapmaximizer_experiment.py --reward-mode normalized_score_diff
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
ROUNDS_PER_RUN = 10
RUNS_PER_ARCHITECTURE = 3


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


def create_double_dqn_gapmaximizer_experiment(
    name: str = "double_dqn_gapmaximizer",
    description: str = "",
    runs_per_arch: int = RUNS_PER_ARCHITECTURE,
    reward_mode: str = "binary",
) -> Path:
    """
    Create an experiment: scale-11 and large_hidden vs GapMaximizer with Double DQN.
    No self-play, no validation. Trainee position alternates each round.
    Reward mode is configurable (binary or normalized_score_diff).
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

    run_order: List[str] = []
    for i in range(runs_per_arch):
        run_order.append(f"scale_11_vs_gapmax_run_{i + 1:02d}")
        run_order.append(f"large_hidden_vs_gapmax_run_{i + 1:02d}")

    base_desc = (
        f"Scale-11 and large_hidden vs GapMaximizer with Double DQN (no self-play), "
        f"{runs_per_arch} seeds each, {ROUNDS_PER_RUN} rounds. "
        "Validation skipped. Trainee position alternates each round."
    )
    if reward_mode != "binary":
        base_desc += f" Reward mode: {reward_mode}."

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or base_desc,
        "experiment_type": "double_dqn_gapmaximizer",
        "train_vs_gapmaximizer": True,
        "skip_validation": True,
        "trainee_first_only": False,
        "use_double_dqn": True,
        "reward_mode": reward_mode,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "run_order": run_order,
        "rounds_per_run": ROUNDS_PER_RUN,
        "runs_per_architecture": runs_per_arch,
        "total_runs": total_runs,
        "seeds": seeds,
    }

    # Copy base config and set Double DQN + GapMaximizer training options
    base_config_src = project_root / "hyperparams_config.json"
    base_config_dst = experiment_path / "base_hyperparams_config.json"
    if base_config_src.exists():
        shutil.copy(base_config_src, base_config_dst)
        with open(base_config_dst) as f:
            config = json.load(f)
        config["use_double_dqn"] = True
        if "training" not in config:
            config["training"] = {}
        config["training"]["train_vs_gapmaximizer"] = True
        config["training"]["skip_validation"] = True
        config["training"]["trainee_first_only"] = False
        config["training"]["reward_mode"] = reward_mode
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
    print(f"Training: vs GapMaximizer (no self-play)")
    print(f"Double DQN: enabled")
    print(f"Validation: skipped")
    print(f"Position: alternates each round (trainee first in even rounds, second in odd)")
    print(f"Reward mode: {reward_mode}")
    print(f"\nArchitectures: scale_11 (game_based), large_hidden")
    print(f"Runs per architecture: {runs_per_arch} (total {total_runs} runs)")
    print(f"Run order:")
    for idx, rid in enumerate(run_order, 1):
        info = runs[rid]
        arch = "game_based " + str(info["hidden_layers"]) if info["network_type"] == "game_based" else "large_hidden [512]"
        print(f"  {idx}. {rid} — {arch}, seed {info['seed']}, {ROUNDS_PER_RUN} rounds")
    print(f"\nRound checkpointing: model_round_0.pt .. model_round_{ROUNDS_PER_RUN - 1}.pt, model_final.pt")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"\nTo use a different reward mode, re-run this script with --reward-mode binary or --reward-mode normalized_score_diff")
    print(f"{'='*70}\n")

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create experiment: scale-11 and large_hidden vs GapMaximizer with Double DQN (reward mode configurable)"
    )
    parser.add_argument("--name", "-n", default="double_dqn_gapmaximizer",
                        help="Experiment name (default: double_dqn_gapmaximizer)")
    parser.add_argument("--description", "-d", default="",
                        help="Experiment description")
    parser.add_argument("--runs-per-arch", "-r", type=int, default=RUNS_PER_ARCHITECTURE,
                        help=f"Runs (seeds) per architecture (default: {RUNS_PER_ARCHITECTURE})")
    parser.add_argument("--reward-mode", "-R", default="binary",
                        choices=["binary", "normalized_score_diff"],
                        help="Reward mode (default: binary)")

    args = parser.parse_args()
    create_double_dqn_gapmaximizer_experiment(
        args.name,
        args.description,
        runs_per_arch=args.runs_per_arch,
        reward_mode=args.reward_mode,
    )
