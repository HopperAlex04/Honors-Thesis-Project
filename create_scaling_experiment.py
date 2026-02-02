#!/usr/bin/env python3
"""
Create an experiment that scales game_based hidden layers to match large_hidden parameters.

This script creates an experiment with:
- Multiple game_based scales (testing different hidden layer sizes)
- Large_hidden baseline for comparison
- All using embedding input
- Multiple runs per configuration for statistical significance
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


def count_params_embedding_network(hidden_layers: List[int]) -> int:
    """Count parameters in embedding-based network."""
    embedding_dim = 52
    zone_encoded_dim = 52
    num_zones = 9
    num_actions = 3157
    fusion_dim = num_zones * zone_encoded_dim  # 468
    
    params = 0
    # Embedding layer
    params += 52 * embedding_dim
    # Zone aggregator
    params += (embedding_dim + 1) * zone_encoded_dim
    # Hidden layers
    prev_dim = fusion_dim
    for hidden_dim in hidden_layers:
        params += (prev_dim + 1) * hidden_dim
        prev_dim = hidden_dim
    # Output layer
    params += (prev_dim + 1) * num_actions
    return params


def find_scales_to_test() -> List[Dict]:
    """Find game_based scales to test, including the one that matches large_hidden."""
    # Calculate large_hidden parameters
    large_hidden_params = count_params_embedding_network([512])
    
    scales_to_test = []
    
    # Test scales from 1 to 25 (or until we exceed large_hidden by a good margin)
    for scale in range(1, 26):
        hidden_layers = [52 * scale, 13 * scale, 15 * scale]
        params = count_params_embedding_network(hidden_layers)
        ratio = params / large_hidden_params
        
        scales_to_test.append({
            "scale": scale,
            "hidden_layers": hidden_layers,
            "params": params,
            "ratio": ratio
        })
        
        # Stop if we've exceeded large_hidden by 20%
        if ratio >= 1.2:
            break
    
    return scales_to_test


def create_scaling_experiment(
    name: str = "game_based_scaling",
    description: str = "",
    baseline_experiment: Optional[str] = None
) -> Path:
    """Create an experiment testing different game_based scales."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Create runs directory
    (experiment_path / "runs").mkdir(exist_ok=True)
    
    # Find scales to test
    scales = find_scales_to_test()
    large_hidden_params = count_params_embedding_network([512])
    
    # Calculate total runs: large_hidden baseline (both inputs) + all game_based scales (both inputs)
    # Using 1 run per scale to quickly identify target scale for full experiment
    runs_per_config = 1
    input_types = ["boolean", "embedding"]
    total_runs = (1 + len(scales)) * runs_per_config * len(input_types)  # large_hidden + game_based scales, both inputs
    
    # Generate seeds
    seeds = generate_seeds(total_runs)
    
    # Create metadata
    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or "Scaling game_based hidden layers to match large_hidden parameters",
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "experiment_type": "scaling",
        "baseline_experiment": baseline_experiment,  # Name of experiment to load baseline from
        "baseline": {
            "network_type": "large_hidden",
            "hidden_layers": [512],
            "params": large_hidden_params
        },
        "scales": [
            {
                "scale": s["scale"],
                "hidden_layers": s["hidden_layers"],
                "params": s["params"],
                "ratio_to_baseline": s["ratio"]
            }
            for s in scales
        ],
        "runs_per_config": runs_per_config,
        "total_runs": total_runs,
        "seeds": seeds
    }
    
    # Copy base config
    base_config = project_root / "hyperparams_config.json"
    if base_config.exists():
        shutil.copy(base_config, experiment_path / "base_hyperparams_config.json")
    
    # Initialize run tracking
    runs = {}
    run_idx = 0
    
    # Add large_hidden baseline runs (both boolean and embedding)
    for input_type in input_types:
        for run_num in range(1, runs_per_config + 1):
            run_id = f"large_hidden_{input_type}_baseline_run_{run_num:02d}"
            runs[run_id] = {
                "run_id": run_id,
                "network_type": "large_hidden",
                "input_type": input_type,
                "scale": None,
                "hidden_layers": [512],
                "params": large_hidden_params,
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
    
    # Add game_based scaling runs (both boolean and embedding)
    for scale_info in scales:
        scale = scale_info["scale"]
        hidden_layers = scale_info["hidden_layers"]
        params = scale_info["params"]
        
        for input_type in input_types:
            for run_num in range(1, runs_per_config + 1):
                run_id = f"game_based_{input_type}_scale_{scale:02d}_run_{run_num:02d}"
                runs[run_id] = {
                    "run_id": run_id,
                    "network_type": "game_based",
                    "input_type": input_type,
                    "scale": scale,
                    "hidden_layers": hidden_layers,
                    "params": params,
                    "ratio_to_baseline": params / large_hidden_params,
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
    
    # Save metadata and runs
    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)
    
    # Save current experiment pointer
    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))
    
    # Print summary
    print(f"{'='*70}")
    print(f"Created scaling experiment: {experiment_name}")
    print(f"{'='*70}\n")
    print(f"Location: {experiment_path}\n")
    print(f"Baseline: large_hidden ([512]) - {large_hidden_params:,} parameters")
    print(f"Input types: {', '.join(input_types)}")
    print(f"\nGame-based scales to test: {len(scales)}")
    print("-" * 70)
    for scale_info in scales:
        print(f"Scale {scale_info['scale']:2d}: {scale_info['hidden_layers']} → "
              f"{scale_info['params']:10,} params ({scale_info['ratio']:.2f}x baseline)")
    print("-" * 70)
    print(f"\nRuns per configuration: {runs_per_config} (screening experiment)")
    print(f"Input types: {len(input_types)} ({', '.join(input_types)})")
    print(f"Total runs: {total_runs}")
    print(f"\nNote: This is a screening experiment to identify target scale.")
    print(f"      Full experiment with multiple runs should be run after identifying target.")
    print(f"{'='*70}\n")
    
    return experiment_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a game_based scaling experiment")
    parser.add_argument("--name", "-n", default="game_based_scaling",
                       help="Experiment name (default: game_based_scaling)")
    parser.add_argument("--description", "-d", default="",
                       help="Experiment description")
    parser.add_argument("--baseline-experiment", "-b", default=None,
                       help="Name of experiment to load baseline win rate from (e.g., experiment_20260123_173115_architecture_comparison)")
    
    args = parser.parse_args()
    
    create_scaling_experiment(args.name, args.description, args.baseline_experiment)
