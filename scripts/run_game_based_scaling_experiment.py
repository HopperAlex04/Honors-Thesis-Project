#!/usr/bin/env python3
"""
Game-Based Scaling Experiment - Scale game-based architecture until it surpasses large_hidden baseline.

This script continuously scales the game-based hidden layers (maintaining card → rank → action-type structure)
until game-based performance surpasses the large hidden layer (512) baseline.

Usage:
    python scripts/run_game_based_scaling_experiment.py
    python scripts/run_game_based_scaling_experiment.py --baseline-experiment path/to/experiment
    python scripts/run_game_based_scaling_experiment.py --max-scale 10 --runs-per-scale 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiment_manager import (
    ExperimentManager,
    format_duration,
    collect_run_metrics,
    generate_seeds,
    get_git_commit,
)


def get_baseline_performance(
    manager: ExperimentManager,
    network_type: str = "large_hidden"
) -> Optional[Tuple[float, float]]:
    """
    Get mean performance and standard error for baseline network type.
    
    Args:
        manager: Experiment manager instance
        network_type: Network type to use as baseline (default: "large_hidden")
        
    Returns:
        Tuple of (mean_win_rate, std_error) or None if insufficient data
    """
    runs = manager.get_runs_by_type(network_type)
    completed_runs = [r for r in runs if r.get("status") == "completed" and r.get("final_metrics")]
    
    if len(completed_runs) < 1:
        return None
    
    win_rates = []
    for run in completed_runs:
        metrics = run.get("final_metrics", {})
        # Try different metric names
        win_rate = (
            metrics.get("final_win_rate") or
            metrics.get("win_rate") or
            metrics.get("validation_win_rate")
        )
        if win_rate is not None:
            win_rates.append(win_rate)
    
    if len(win_rates) < 1:
        return None
    
    import numpy as np
    mean_win_rate = np.mean(win_rates)
    std_error = np.std(win_rates, ddof=1) / np.sqrt(len(win_rates)) if len(win_rates) > 1 else 0.0
    
    return mean_win_rate, std_error


def run_scale_experiment(
    manager: ExperimentManager,
    scale: int,
    runs_per_scale: int,
    project_root: Path
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Run game-based experiment at a specific scale.
    
    Args:
        manager: Experiment manager instance
        scale: Scale factor k (uses [52*k, 13*k, 15*k])
        runs_per_scale: Number of runs to execute at this scale
        project_root: Path to project root
        
    Returns:
        Tuple of (success, mean_win_rate, error_message)
    """
    print(f"\n{'='*70}")
    print(f"Running scale k={scale} (hidden_layers=[{52*scale}, {13*scale}, {15*scale}])")
    print(f"{'='*70}")
    
    # Create experiment for this scale if it doesn't exist
    scale_experiment_name = f"game_based_scale_{scale}"
    scale_experiment_path = manager.experiment_path.parent / scale_experiment_name
    
    if not scale_experiment_path.exists():
        # Initialize new experiment for this scale
        # We'll create a minimal experiment with just game_based runs
        scale_experiment_path.mkdir(parents=True, exist_ok=True)
        (scale_experiment_path / "runs").mkdir(exist_ok=True)
        
        # Create metadata
        from datetime import datetime
        
        seeds = generate_seeds(runs_per_scale)
        metadata = {
            "experiment_name": scale_experiment_name,
            "display_name": f"Game-based Scale {scale}",
            "description": f"Game-based scaling experiment at scale k={scale}",
            "created_at": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "network_types": ["game_based"],
            "runs_per_type": runs_per_scale,
            "total_runs": runs_per_scale,
            "status": "initialized",
            "seeds": seeds,
        }
        
        with open(scale_experiment_path / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create runs
        runs = {}
        for run_num in range(1, runs_per_scale + 1):
            run_id = f"game_based_run_{run_num:02d}"
            runs[run_id] = {
                "run_id": run_id,
                "network_type": "game_based",
                "run_number": run_num,
                "seed": seeds[run_num - 1],
                "status": "pending",
                "created_at": None,
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "final_metrics": None,
                "error_message": None,
            }
        
        with open(scale_experiment_path / "runs_status.json", 'w') as f:
            json.dump(runs, f, indent=2)
        
        scale_manager = ExperimentManager(scale_experiment_path)
    else:
        scale_manager = ExperimentManager(scale_experiment_path)
    
    # Update base config with scale
    base_config = scale_manager.experiment_path / "base_hyperparams_config.json"
    if base_config.exists():
        with open(base_config, 'r') as f:
            config = json.load(f)
    else:
        with open(project_root / "hyperparams_config.json", 'r') as f:
            config = json.load(f)
    
    config["network_type"] = "game_based"
    config["game_based_scale"] = scale
    config["game_based_hidden_layers"] = None  # Use scale-based calculation
    
    with open(base_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get runs for this scale
    scale_runs = scale_manager.get_runs_by_type("game_based")
    pending_runs = [r for r in scale_runs if r.get("status") not in ["completed", "running"]]
    
    # Run up to runs_per_scale runs
    runs_to_execute = pending_runs[:runs_per_scale]
    
    if not runs_to_execute:
        print(f"All runs for scale {scale} already completed.")
        # Collect existing metrics
        completed = [r for r in scale_runs if r.get("status") == "completed"]
        if completed:
            win_rates = []
            for run in completed:
                metrics = run.get("final_metrics", {})
                win_rate = (
                    metrics.get("final_win_rate") or
                    metrics.get("win_rate") or
                    metrics.get("validation_win_rate")
                )
                if win_rate is not None:
                    win_rates.append(win_rate)
            
            if win_rates:
                import numpy as np
                mean_win_rate = np.mean(win_rates)
                return True, mean_win_rate, None
    
    # Execute runs using run_full_experiment logic
    from scripts.run_full_experiment import run_single_training
    
    results = {}
    for run_info in runs_to_execute:
        run_id = run_info["run_id"]
        run_path = scale_manager.get_run_path(run_id)
        
        scale_manager.prepare_run(run_id)
        scale_manager.start_run(run_id)
        
        _, success, metrics, error = run_single_training(
            run_id,
            run_path,
            run_info["seed"],
            "game_based",
            project_root,
        )
        
        scale_manager.complete_run(run_id, metrics=metrics, error=error)
        results[run_id] = success
    
    # Collect metrics
    completed_runs = [r for r in scale_manager.get_runs_by_type("game_based") 
                      if r.get("status") == "completed"]
    
    if len(completed_runs) < 1:
        return False, None, "No completed runs"
    
    win_rates = []
    for run in completed_runs:
        metrics = run.get("final_metrics", {})
        win_rate = (
            metrics.get("final_win_rate") or
            metrics.get("win_rate") or
            metrics.get("validation_win_rate")
        )
        if win_rate is not None:
            win_rates.append(win_rate)
    
    if len(win_rates) < 1:
        return False, None, "No valid win rate metrics found"
    
    import numpy as np
    mean_win_rate = np.mean(win_rates)
    
    return True, mean_win_rate, None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scale game-based architecture until it surpasses large_hidden baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--baseline-experiment",
        type=str,
        help="Path to experiment containing large_hidden baseline results"
    )
    parser.add_argument(
        "--max-scale",
        type=int,
        default=10,
        help="Maximum scale factor to try (default: 10)"
    )
    parser.add_argument(
        "--runs-per-scale",
        type=int,
        default=7,
        help="Number of runs per scale (default: 7)"
    )
    parser.add_argument(
        "--init",
        metavar="NAME",
        help="Initialize a new scaling experiment with this name"
    )
    
    args = parser.parse_args()
    
    # Get or create experiment
    if args.init:
        manager = ExperimentManager.init_experiment(args.init, description="Game-based scaling experiment")
    elif args.baseline_experiment:
        baseline_path = Path(args.baseline_experiment)
        if not baseline_path.exists():
            print(f"Error: Baseline experiment path does not exist: {baseline_path}")
            return 1
        manager = ExperimentManager(baseline_path)
    else:
        manager = ExperimentManager.get_current()
    
    if not manager:
        print("No current experiment. Use --init NAME or --baseline-experiment PATH")
        return 1
    
    # Get baseline performance
    print("\n" + "="*70)
    print("Getting large_hidden baseline performance...")
    print("="*70)
    
    baseline_perf = get_baseline_performance(manager, "large_hidden")
    
    if baseline_perf is None:
        print("Warning: No large_hidden baseline found. Running baseline first...")
        # TODO: Could run baseline here, but for now just warn
        print("Please run large_hidden experiments first or provide --baseline-experiment")
        return 1
    
    baseline_mean, baseline_std = baseline_perf
    print(f"Baseline (large_hidden): {baseline_mean:.1%} ± {baseline_std:.1%}")
    
    # Scaling loop
    print("\n" + "="*70)
    print("Starting scaling experiment...")
    print("="*70)
    
    results = []
    scale = 1
    
    while scale <= args.max_scale:
        success, mean_win_rate, error = run_scale_experiment(
            manager, scale, args.runs_per_scale, project_root
        )
        
        if not success:
            print(f"Scale {scale} failed: {error}")
            scale += 1
            continue
        
        if mean_win_rate is None:
            print(f"Scale {scale}: No valid metrics")
            scale += 1
            continue
        
        # Calculate parameter count
        from src.cuttle.network_dimensions import count_parameters_boolean_network
        params = count_parameters_boolean_network(
            468, 3157, tuple([52*scale, 13*scale, 15*scale])
        )
        
        results.append({
            "scale": scale,
            "hidden_layers": [52*scale, 13*scale, 15*scale],
            "mean_win_rate": mean_win_rate,
            "parameters": params,
            "surpasses_baseline": mean_win_rate > baseline_mean
        })
        
        print(f"\nScale k={scale}: Win rate = {mean_win_rate:.1%}, Params = {params:,}")
        print(f"  Baseline: {baseline_mean:.1%}")
        print(f"  Surpasses: {'YES' if mean_win_rate > baseline_mean else 'NO'}")
        
        # Check if we've surpassed baseline
        if mean_win_rate > baseline_mean:
            print(f"\n{'='*70}")
            print(f"SUCCESS: Game-based (k={scale}) surpasses large_hidden baseline!")
            print(f"  Game-based: {mean_win_rate:.1%}")
            print(f"  Baseline:   {baseline_mean:.1%}")
            print(f"  Parameters: {params:,}")
            print(f"{'='*70}")
            break
        
        scale += 1
    
    # Summary table
    print("\n" + "="*70)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Scale':<8} {'Layers':<25} {'Win Rate':<12} {'Params':<12} {'Surpasses':<10}")
    print("-" * 70)
    
    for r in results:
        layers_str = str(r["hidden_layers"])
        win_rate_str = f"{r['mean_win_rate']:.1%}"
        params_str = f"{r['parameters']:,}"
        surpasses_str = "YES" if r["surpasses_baseline"] else "NO"
        print(f"{r['scale']:<8} {layers_str:<25} {win_rate_str:<12} {params_str:<12} {surpasses_str:<10}")
    
    if results and not any(r["surpasses_baseline"] for r in results):
        print(f"\nNo scale up to k={args.max_scale} surpassed baseline.")
        print(f"Consider increasing --max-scale or investigating architecture.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
