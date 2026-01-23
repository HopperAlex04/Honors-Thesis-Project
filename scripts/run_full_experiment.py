#!/usr/bin/env python3
"""
Run Full Experiment - Execute all training runs (15 total: 3 types × 5 runs).

This script automates running all training runs for the architecture comparison
experiment. It can run sequentially or in parallel.

Usage:
    python scripts/run_full_experiment.py                  # Run all pending, sequential
    python scripts/run_full_experiment.py --parallel 3     # Run up to 3 in parallel
    python scripts/run_full_experiment.py --type linear    # Run only linear network runs
    python scripts/run_full_experiment.py --dry-run        # Show what would be run
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiment_manager import (
    ExperimentManager,
    NETWORK_TYPES,
    RUNS_PER_TYPE,
    format_duration,
    collect_run_metrics,
)


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_single_training(
    run_id: str,
    run_path: Path,
    seed: int,
    network_type: str,
    project_root: Path,
) -> Tuple[str, bool, Optional[Dict], Optional[str]]:
    """
    Execute a single training run in an isolated manner.
    
    Args:
        run_id: Run identifier
        run_path: Path to run directory
        seed: Random seed for this run
        network_type: Network type to use
        project_root: Path to project root
        
    Returns:
        Tuple of (run_id, success, metrics, error_message)
    """
    print(f"\n[{run_id}] Starting training...")
    start_time = time.time()
    
    try:
        # Set up the run directory
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "models").mkdir(exist_ok=True)
        (run_path / "action_logs").mkdir(exist_ok=True)
        
        # Load and modify config
        config_file = run_path / "hyperparams_config.json"
        if not config_file.exists():
            base_config = run_path.parent.parent / "base_hyperparams_config.json"
            if base_config.exists():
                with open(base_config, 'r') as f:
                    config = json.load(f)
            else:
                # Use project default
                with open(project_root / "hyperparams_config.json", 'r') as f:
                    config = json.load(f)
            
            config["network_type"] = network_type
            config["random_seed"] = seed
            if "training" in config:
                config["training"]["quick_test_mode"] = False
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create a temporary working directory for this run
        work_dir = run_path / "workspace"
        work_dir.mkdir(exist_ok=True)
        
        # Copy necessary files
        shutil.copy(project_root / "train.py", work_dir / "train.py")
        shutil.copytree(project_root / "src", work_dir / "src", dirs_exist_ok=True)
        shutil.copy(config_file, work_dir / "hyperparams_config.json")
        
        # Create models and log directories in workspace
        (work_dir / "models").mkdir(exist_ok=True)
        (work_dir / "action_logs").mkdir(exist_ok=True)
        (work_dir / "metrics_logs").mkdir(exist_ok=True)
        
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(work_dir / "src")
        
        # Run training
        result = subprocess.run(
            [sys.executable, str(work_dir / "train.py")],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Copy results back
        if (work_dir / "models").exists():
            shutil.copytree(work_dir / "models", run_path / "models", dirs_exist_ok=True)
        if (work_dir / "action_logs").exists():
            shutil.copytree(work_dir / "action_logs", run_path / "action_logs", dirs_exist_ok=True)
        if (work_dir / "metrics_logs").exists():
            shutil.copytree(work_dir / "metrics_logs", run_path / "metrics_logs", dirs_exist_ok=True)
        
        # Clean up workspace
        shutil.rmtree(work_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = f"Training failed (exit {result.returncode}): {result.stderr[-500:] if result.stderr else 'No error output'}"
            print(f"[{run_id}] Failed after {format_duration(elapsed)}: {error_msg[:100]}...")
            return run_id, False, None, error_msg
        
        # Collect metrics
        metrics = collect_run_metrics(run_path)
        
        print(f"[{run_id}] Completed in {format_duration(elapsed)}")
        if metrics and "final_win_rate" in metrics:
            print(f"[{run_id}] Final win rate: {metrics['final_win_rate']:.1%}")
        
        return run_id, True, metrics, None
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[{run_id}] Error after {format_duration(elapsed)}: {error_msg}")
        return run_id, False, None, error_msg


def run_experiment_sequential(
    manager: ExperimentManager,
    run_ids: List[str],
) -> Dict[str, bool]:
    """Run experiments sequentially."""
    results = {}
    
    total = len(run_ids)
    for i, run_id in enumerate(run_ids, 1):
        print(f"\n{'='*70}")
        print(f"Run {i}/{total}: {run_id}")
        print(f"{'='*70}")
        
        run_info = manager.runs[run_id]
        run_path = manager.get_run_path(run_id)
        
        # Prepare run
        manager.prepare_run(run_id)
        manager.start_run(run_id)
        
        # Execute training
        _, success, metrics, error = run_single_training(
            run_id,
            run_path,
            run_info["seed"],
            run_info["network_type"],
            project_root,
        )
        
        # Update status
        manager.complete_run(run_id, metrics=metrics, error=error)
        results[run_id] = success
    
    return results


def run_experiment_parallel(
    manager: ExperimentManager,
    run_ids: List[str],
    max_workers: int,
) -> Dict[str, bool]:
    """Run experiments in parallel."""
    results = {}
    
    # Prepare all runs first
    prepared_runs = []
    for run_id in run_ids:
        run_path = manager.prepare_run(run_id)
        run_info = manager.runs[run_id]
        prepared_runs.append((
            run_id,
            run_path,
            run_info["seed"],
            run_info["network_type"],
        ))
    
    print(f"\nStarting {len(prepared_runs)} runs with {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {}
        for run_id, run_path, seed, network_type in prepared_runs:
            manager.start_run(run_id)
            future = executor.submit(
                run_single_training,
                run_id,
                run_path,
                seed,
                network_type,
                project_root,
            )
            futures[future] = run_id
        
        # Collect results as they complete
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                _, success, metrics, error = future.result()
                manager.complete_run(run_id, metrics=metrics, error=error)
                results[run_id] = success
            except Exception as e:
                manager.complete_run(run_id, error=str(e))
                results[run_id] = False
    
    return results


def print_summary(manager: ExperimentManager, results: Dict[str, bool]) -> None:
    """Print a summary of the experiment run."""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"\nRuns completed: {succeeded}")
    print(f"Runs failed: {failed}")
    
    if failed > 0:
        print("\nFailed runs:")
        for run_id, success in results.items():
            if not success:
                error = manager.runs[run_id].get("error_message", "Unknown error")
                print(f"  - {run_id}: {error[:80]}...")
    
    # Print overall status
    print()
    manager.print_status()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full experiment (all 21 training runs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel runs (default: 1, sequential)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=NETWORK_TYPES,
        help="Only run specific network type"
    )
    parser.add_argument(
        "--runs", "-r",
        help="Specific run numbers to execute (comma-separated, e.g., '1,2,3')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if already completed"
    )
    parser.add_argument(
        "--init",
        metavar="NAME",
        help="Initialize a new experiment with this name first"
    )
    
    args = parser.parse_args()
    
    # Initialize experiment if requested
    if args.init:
        manager = ExperimentManager.init_experiment(args.init)
    else:
        manager = ExperimentManager.get_current()
    
    if not manager:
        print("No current experiment. Use --init NAME or run 'experiment_manager.py init' first.")
        return 1
    
    # Determine which runs to execute
    all_run_ids = list(manager.runs.keys())
    
    if args.type:
        all_run_ids = [r for r in all_run_ids if manager.runs[r]["network_type"] == args.type]
    
    if args.runs:
        run_nums = [int(x) for x in args.runs.split(",")]
        all_run_ids = [r for r in all_run_ids if manager.runs[r]["run_number"] in run_nums]
    
    # Filter out completed runs unless --force
    if not args.force:
        run_ids = [r for r in all_run_ids if manager.runs[r]["status"] not in ["completed"]]
    else:
        run_ids = all_run_ids
    
    if not run_ids:
        print("No runs to execute (all completed or none match criteria).")
        manager.print_status()
        return 0
    
    # Show what would be run
    print(f"\nRuns to execute ({len(run_ids)}):")
    for run_id in run_ids:
        info = manager.runs[run_id]
        print(f"  - {run_id} (seed: {info['seed']}, status: {info['status']})")
    
    if args.dry_run:
        print("\n[DRY RUN] No runs executed.")
        return 0
    
    # Confirm
    print(f"\nThis will run {len(run_ids)} training sessions.")
    if args.parallel > 1:
        print(f"Running up to {args.parallel} in parallel.")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Cancelled.")
        return 0
    
    # Execute
    start_time = time.time()
    
    if args.parallel > 1:
        results = run_experiment_parallel(manager, run_ids, args.parallel)
    else:
        results = run_experiment_sequential(manager, run_ids)
    
    elapsed = time.time() - start_time
    
    # Summary
    print_summary(manager, results)
    print(f"\nTotal time: {format_duration(elapsed)}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
