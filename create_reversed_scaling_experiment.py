#!/usr/bin/env python3
"""
Create an experiment with reversed (flipped) game_based layer order: [15k, 13k, 52k].

Unlike the original [52k, 13k, 15k] (wide→narrow), the reversed order is narrow→wide,
with the 52-card representation at the end before the output layer.

Scales start at max k and work DOWN to k=1 for both boolean and embedding inputs.
This tests whether the reversed order can achieve comparable performance with
fewer parameters, or perform better at the same parameter budget.

Default max_scale=25 (mirrors original scaling experiment); flipped [15k, 13k, 52k]
matches large_hidden at k≈10-11.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Project root
project_root = Path(__file__).parent
experiments_dir = project_root / "experiments"

# Constants
NUM_ACTIONS = 3157
INPUT_DIM = 468  # Boolean and embedding fusion_dim
EMBEDDING_DIM = 52
ZONE_ENCODED_DIM = 52
NUM_ZONES = 9
FUSION_DIM = NUM_ZONES * ZONE_ENCODED_DIM  # 468


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


def flipped_hidden_layers(scale: int) -> List[int]:
    """Reversed order: [15k, 13k, 52k] (narrow → wide)."""
    return [15 * scale, 13 * scale, 52 * scale]


def count_params_boolean(hidden_layers: List[int]) -> int:
    """Count parameters for boolean network (no embedding)."""
    params = 0
    prev_dim = INPUT_DIM
    for h in hidden_layers:
        params += (prev_dim + 1) * h
        prev_dim = h
    params += (prev_dim + 1) * NUM_ACTIONS
    return params


def count_params_embedding(hidden_layers: List[int]) -> int:
    """Count parameters for embedding network."""
    params = 52 * EMBEDDING_DIM + (EMBEDDING_DIM + 1) * ZONE_ENCODED_DIM
    prev_dim = FUSION_DIM
    for h in hidden_layers:
        params += (prev_dim + 1) * h
        prev_dim = h
    params += (prev_dim + 1) * NUM_ACTIONS
    return params


def find_reversed_scales_descending(max_scale: int = 25) -> List[Dict]:
    """
    Find scales to test: start at max_scale, work DOWN to 1.
    Returns scales in descending order (highest first).
    """
    large_hidden_params = count_params_embedding([512])

    # Build scales from max_scale down to 1
    scales_to_test = []
    for k in range(max_scale, 0, -1):
        layers = flipped_hidden_layers(k)
        params_emb = count_params_embedding(layers)
        params_bool = count_params_boolean(layers)
        ratio_emb = params_emb / large_hidden_params
        ratio_bool = params_bool / large_hidden_params

        scales_to_test.append({
            "scale": k,
            "hidden_layers": layers,
            "params_embedding": params_emb,
            "params_boolean": params_bool,
            "ratio_to_baseline_emb": ratio_emb,
            "ratio_to_baseline_bool": ratio_bool,
        })

    return scales_to_test


def create_reversed_scaling_experiment(
    name: str = "game_based_reversed_scaling",
    description: str = "",
    baseline_experiment: Optional[str] = None,
    max_scale: int = 25
) -> Path:
    """Create an experiment with reversed layer order [15k, 13k, 52k], scaling from max_scale down to 1."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    (experiment_path / "runs").mkdir(exist_ok=True)

    scales = find_reversed_scales_descending(max_scale=max_scale)
    large_hidden_params = count_params_embedding([512])

    runs_per_config = 1
    input_types = ["boolean", "embedding"]
    total_runs = len(scales) * runs_per_config * len(input_types)

    seeds = generate_seeds(total_runs)

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or (
            "Reversed game_based layer order [15k, 13k, 52k] scaling from max k down to 1. "
            "Tests narrow→wide architecture vs original wide→narrow."
        ),
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "experiment_type": "scaling",
        "layer_order": "reversed",
        "hidden_layers_formula": "[15*k, 13*k, 52*k]",
        "scaling_direction": "descending",
        "scale_order": "descending",
        "max_scale": max_scale,
        "baseline_experiment": baseline_experiment,
        "baseline": {
            "reference": "large_hidden",
            "hidden_layers": [512],
            "params": large_hidden_params
        },
        "scales": [
            {
                "scale": s["scale"],
                "hidden_layers": s["hidden_layers"],
                "params_embedding": s["params_embedding"],
                "params_boolean": s["params_boolean"],
                "ratio_to_baseline_emb": s["ratio_to_baseline_emb"],
                "ratio_to_baseline_bool": s["ratio_to_baseline_bool"],
            }
            for s in scales
        ],
        "runs_per_config": runs_per_config,
        "input_types": input_types,
        "total_runs": total_runs,
        "seeds": seeds
    }

    base_config = project_root / "hyperparams_config.json"
    if base_config.exists():
        shutil.copy(base_config, experiment_path / "base_hyperparams_config.json")

    runs = {}
    run_idx = 0

    for scale_info in scales:
        scale = scale_info["scale"]
        hidden_layers = scale_info["hidden_layers"]

        for input_type in input_types:
            params = scale_info["params_embedding"] if input_type == "embedding" else scale_info["params_boolean"]
            ratio = scale_info["ratio_to_baseline_emb"] if input_type == "embedding" else scale_info["ratio_to_baseline_bool"]

            for run_num in range(1, runs_per_config + 1):
                run_id = f"game_based_reversed_{input_type}_scale_{scale:02d}_run_{run_num:02d}"
                runs[run_id] = {
                    "run_id": run_id,
                    "network_type": "game_based",
                    "input_type": input_type,
                    "scale": scale,
                    "hidden_layers": hidden_layers,
                    "params": params,
                    "ratio_to_baseline": ratio,
                    "run_number": run_num,
                    "seed": seeds[run_idx],
                    "status": "pending",
                    "created_at": None,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "final_metrics": None,
                    "error_message": None
                }
                run_idx += 1

    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))

    # Print summary
    print(f"{'='*70}")
    print(f"Created REVERSED scaling experiment: {experiment_name}")
    print(f"{'='*70}\n")
    print(f"Location: {experiment_path}\n")
    print(f"Layer order: [15k, 13k, 52k] (narrow → wide, reversed from original)")
    print(f"Reference: large_hidden ([512]) = {large_hidden_params:,} params")
    print(f"Scaling: DESCENDING from max k={max_scale} down to k=1\n")
    print(f"Scales to test: {len(scales)}")
    print("-" * 70)
    for s in scales:
        print(f"  k={s['scale']:2d}: {s['hidden_layers']} → "
              f"emb={s['params_embedding']:>9,} bool={s['params_boolean']:>9,} "
              f"({s['ratio_to_baseline_emb']:.2f}x / {s['ratio_to_baseline_bool']:.2f}x)")
    print("-" * 70)
    print(f"\nRuns per configuration: {runs_per_config}")
    print(f"Input types: {input_types}")
    print(f"Total runs: {total_runs}")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"{'='*70}\n")

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create reversed [15k, 13k, 52k] scaling experiment (descending from large_hidden match)"
    )
    parser.add_argument("--name", "-n", default="game_based_reversed_scaling",
                        help="Experiment name (default: game_based_reversed_scaling)")
    parser.add_argument("--description", "-d", default="",
                        help="Experiment description")
    parser.add_argument("--baseline-experiment", "-b", default=None,
                        help="Name of experiment to load baseline win rate from")
    parser.add_argument("--max-scale", "-m", type=int, default=25,
                        help="Maximum scale k; scales run from k=max_scale down to 1 (default: 25)")

    args = parser.parse_args()

    create_reversed_scaling_experiment(
        args.name,
        args.description,
        args.baseline_experiment,
        args.max_scale
    )
