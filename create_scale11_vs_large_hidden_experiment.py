#!/usr/bin/env python3
"""
Create an experiment that runs scale-11 game_based and large_hidden for 10 rounds each,
with multiple seeds per architecture for statistical comparison.

Each run uses round checkpointing so the best model by validation (vs GapMaximizer)
can be used later instead of only the final model.

Usage:
    python create_scale11_vs_large_hidden_experiment.py
    python create_scale11_vs_large_hidden_experiment.py --name my_comparison
    python create_scale11_vs_large_hidden_experiment.py --runs-per-arch 3
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
# Number of runs (seeds) per architecture for variance / significance
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


def create_scale11_vs_large_hidden_experiment(
    name: str = "scale11_vs_large_hidden_10rounds",
    description: str = "",
    runs_per_arch: int = RUNS_PER_ARCHITECTURE,
) -> Path:
    """
    Create an experiment with multiple runs per architecture (scale_11 and large_hidden),
    alternating order so we get early comparison across seeds.

    Each run gets 10 training rounds. Round checkpointing is enabled in train.py.
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
        run_id = f"scale_11_embedding_run_{i + 1:02d}"
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
        run_id = f"large_hidden_embedding_run_{i + 1:02d}"
        runs[run_id] = _run_template(
            run_id=run_id,
            run_number=run_num,
            seed=seeds[2 * i + 1],
            network_type="large_hidden",
            input_type="embedding",
            scale=None,
            hidden_layers=[512],
        )

    # Alternate order: s11_01, lh_01, s11_02, lh_02, s11_03, lh_03
    run_order_alternating: List[str] = []
    for i in range(runs_per_arch):
        run_order_alternating.append(f"scale_11_embedding_run_{i + 1:02d}")
        run_order_alternating.append(f"large_hidden_embedding_run_{i + 1:02d}")

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or (
            f"Scale-11 game_based vs large_hidden, {runs_per_arch} seeds each, 10 rounds per run. "
            "Round checkpointing enabled: use model_best.pt or model_round_*.pt per run."
        ),
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "experiment_type": "comparison",
        "run_order": run_order_alternating,
        "rounds_per_run": ROUNDS_PER_RUN,
        "runs_per_architecture": runs_per_arch,
        "total_runs": total_runs,
        "seeds": seeds,
    }

    base_config = project_root / "hyperparams_config.json"
    if base_config.exists():
        shutil.copy(base_config, experiment_path / "base_hyperparams_config.json")

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
    print(f"Runs per architecture: {runs_per_arch} (total {total_runs} runs)")
    print(f"Run order (alternating):")
    for idx, rid in enumerate(run_order_alternating, 1):
        info = runs[rid]
        arch = "game_based " + str(info["hidden_layers"]) if info["network_type"] == "game_based" else "large_hidden [512]"
        print(f"  {idx}. {rid} — {arch}, seed {info['seed']}, {ROUNDS_PER_RUN} rounds")
    print(f"\nRound checkpointing: each run saves model_round_0.pt .. model_round_{ROUNDS_PER_RUN - 1}.pt")
    print(f"                    and model_best.pt (best by vs GapMaximizer) + best_round.json")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"{'='*70}\n")

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create scale-11 vs large_hidden experiment (multiple seeds per architecture)"
    )
    parser.add_argument("--name", "-n", default="scale11_vs_large_hidden_10rounds",
                        help="Experiment name (default: scale11_vs_large_hidden_10rounds)")
    parser.add_argument("--description", "-d", default="",
                        help="Experiment description")
    parser.add_argument("--runs-per-arch", "-r", type=int, default=RUNS_PER_ARCHITECTURE,
                        help=f"Runs (seeds) per architecture (default: {RUNS_PER_ARCHITECTURE})")

    args = parser.parse_args()
    create_scale11_vs_large_hidden_experiment(
        args.name,
        args.description,
        runs_per_arch=args.runs_per_arch,
    )
