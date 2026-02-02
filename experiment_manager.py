#!/usr/bin/env python3
"""
Simple experiment manager for running architecture comparisons.

Runs 6 configurations sequentially:
- linear with boolean input
- linear with embedding input
- large_hidden with boolean input
- large_hidden with embedding input
- game_based with boolean input
- game_based with embedding input

Usage:
    python experiment_manager.py init --name "experiment_name"
    python experiment_manager.py run
    python experiment_manager.py run --network linear --input boolean
    python experiment_manager.py run -N large_hidden -N game_based -I embedding
    python experiment_manager.py status
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project root
project_root = Path(__file__).parent
experiments_dir = project_root / "experiments"

# Experiment configurations
NETWORK_TYPES = ["linear", "large_hidden", "game_based"]
INPUT_TYPES = ["boolean", "embedding"]
RUNS_PER_CONFIG = 5  # Number of runs per configuration

TOTAL_RUNS = len(NETWORK_TYPES) * len(INPUT_TYPES) * RUNS_PER_CONFIG


def get_git_commit() -> Optional[str]:
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
    return None


def generate_seeds(count: int, base_seed: int = 42) -> List[int]:
    """Generate deterministic random seeds for reproducibility."""
    np.random.seed(base_seed)
    return [int(np.random.randint(1, 100000)) for _ in range(count)]


def init_experiment(name: str, description: str = "") -> Path:
    """Initialize a new experiment directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Create runs directory
    (experiment_path / "runs").mkdir(exist_ok=True)
    
    # Generate seeds for all runs
    seeds = generate_seeds(TOTAL_RUNS)
    
    # Create metadata
    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "network_types": NETWORK_TYPES,
        "input_types": INPUT_TYPES,
        "runs_per_config": RUNS_PER_CONFIG,
        "total_runs": TOTAL_RUNS,
        "seeds": seeds
    }
    
    # Copy base config
    base_config = project_root / "hyperparams_config.json"
    if base_config.exists():
        shutil.copy(base_config, experiment_path / "base_hyperparams_config.json")
    
    # Initialize run tracking
    runs = {}
    run_idx = 0
    
    for network_type in NETWORK_TYPES:
        for input_type in INPUT_TYPES:
            for run_num in range(1, RUNS_PER_CONFIG + 1):
                run_id = f"{network_type}_{input_type}_run_{run_num:02d}"
                runs[run_id] = {
                    "run_id": run_id,
                    "network_type": network_type,
                    "input_type": input_type,
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
    
    print(f"Initialized experiment: {experiment_name}")
    print(f"Location: {experiment_path}")
    print(f"Configurations: {len(NETWORK_TYPES) * len(INPUT_TYPES)} ({len(NETWORK_TYPES)} networks × {len(INPUT_TYPES)} input types)")
    print(f"Runs per configuration: {RUNS_PER_CONFIG}")
    print(f"Total runs: {TOTAL_RUNS}")
    
    return experiment_path


def get_current_experiment() -> Optional[Path]:
    """Get the current experiment path."""
    current_file = experiments_dir / ".current_experiment"
    if not current_file.exists():
        return None
    
    with open(current_file) as f:
        experiment_path = Path(f.read().strip())
    
    if not experiment_path.exists():
        return None
    
    return experiment_path


def load_experiment(experiment_path: Path) -> tuple:
    """Load experiment metadata and runs."""
    with open(experiment_path / "experiment_metadata.json") as f:
        metadata = json.load(f)
    
    with open(experiment_path / "runs_status.json") as f:
        runs = json.load(f)
    
    return metadata, runs


def save_runs(experiment_path: Path, runs: Dict) -> None:
    """Save runs status."""
    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)


def create_run_config(experiment_path: Path, run_info: Dict) -> Path:
    """Create hyperparameter config for a specific run."""
    run_id = run_info["run_id"]
    run_path = experiment_path / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for this run
    (run_path / "models").mkdir(exist_ok=True)
    (run_path / "action_logs").mkdir(exist_ok=True)
    (run_path / "metrics_logs").mkdir(exist_ok=True)
    
    # Load base config
    base_config_path = experiment_path / "base_hyperparams_config.json"
    with open(base_config_path) as f:
        config = json.load(f)
    
    # Set network type and embedding flag
    config["network_type"] = run_info["network_type"]
    config["use_embeddings"] = (run_info["input_type"] == "embedding")
    config["random_seed"] = run_info["seed"]
    
    # Handle scaling experiments: set game_based_scale or game_based_hidden_layers
    if run_info["network_type"] == "game_based" and "scale" in run_info and run_info["scale"] is not None:
        # Scaling experiment: set explicit hidden layers
        if "hidden_layers" in run_info:
            config["game_based_hidden_layers"] = run_info["hidden_layers"]
            # Remove game_based_scale if present to avoid confusion
            if "game_based_scale" in config:
                del config["game_based_scale"]
        else:
            # Fallback to scale
            config["game_based_scale"] = run_info["scale"]
    
    # Save run config
    with open(run_path / "hyperparams_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create run metadata
    run_metadata = {
        "run_id": run_id,
        "network_type": run_info["network_type"],
        "input_type": run_info["input_type"],
        "run_number": run_info["run_number"],
        "seed": run_info["seed"],
        "experiment_name": experiment_path.name,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    with open(run_path / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    return run_path


def run_single_training(
    experiment_path: Path,
    run_info: Dict,
    run_path: Path
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Execute a single training run."""
    run_id = run_info["run_id"]
    print(f"\n{'='*70}")
    print(f"Running: {run_id}")
    print(f"  Network: {run_info['network_type']}")
    print(f"  Input: {run_info['input_type']}")
    if "scale" in run_info and run_info["scale"] is not None:
        print(f"  Scale: {run_info['scale']}")
        if "hidden_layers" in run_info:
            print(f"  Hidden layers: {run_info['hidden_layers']}")
    print(f"  Seed: {run_info['seed']}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Create workspace for this run
        work_dir = run_path / "workspace"
        work_dir.mkdir(exist_ok=True)
        
        # Copy necessary files
        shutil.copy(project_root / "train.py", work_dir / "train.py")
        shutil.copytree(project_root / "src", work_dir / "src", dirs_exist_ok=True)
        shutil.copy(run_path / "hyperparams_config.json", work_dir / "hyperparams_config.json")
        
        # Create output directories in workspace
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
        
        # Copy results back to run directory
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
            error_msg = f"Training failed (exit {result.returncode})"
            if result.stderr:
                error_msg += f": {result.stderr[-500:]}"
            print(f"[{run_id}] Failed after {elapsed:.1f}s")
            return False, None, error_msg
        
        # Extract final metrics from metrics logs
        metrics = extract_final_metrics(run_path)
        
        print(f"[{run_id}] Completed in {elapsed:.1f}s")
        if metrics and "final_win_rate" in metrics:
            print(f"[{run_id}] Final win rate: {metrics['final_win_rate']:.1%}")
        
        return True, metrics, None
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[{run_id}] Error after {elapsed:.1f}s: {error_msg}")
        return False, None, error_msg


def extract_final_metrics(run_path: Path) -> Optional[Dict]:
    """Extract final metrics from metrics logs."""
    metrics_logs_dir = run_path / "metrics_logs"
    if not metrics_logs_dir.exists():
        return None
    
    # Find the latest round metrics file
    metrics_files = list(metrics_logs_dir.glob("metrics_round_*_vs_*.jsonl"))
    if not metrics_files:
        return None
    
    # Get the highest round number
    rounds = []
    for f in metrics_files:
        # Extract round number from filename like "metrics_round_19_vs_randomized_trainee_first.jsonl"
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[1] == "round":
            try:
                rounds.append(int(parts[2]))
            except ValueError:
                pass
    
    if not rounds:
        return None
    
    max_round = max(rounds)
    
    # Read the latest metrics file
    latest_files = [f for f in metrics_files if f"round_{max_round}_" in f.name]
    if not latest_files:
        return None
    
    # Read the last line (most recent episode)
    latest_file = latest_files[0]
    try:
        with open(latest_file) as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    data = json.loads(last_line)
                    return {
                        "final_win_rate": data.get("p1_win_rate", 0.0),
                        "final_episode": data.get("episode", 0)
                    }
    except Exception:
        pass
    
    return None


def get_baseline_win_rate_from_experiment(baseline_experiment_path: Path, input_type: str = "embedding") -> Optional[float]:
    """Get average win rate for large_hidden baseline runs from another experiment."""
    if not baseline_experiment_path.exists():
        return None
    
    runs_status_file = baseline_experiment_path / "runs_status.json"
    if not runs_status_file.exists():
        return None
    
    with open(runs_status_file) as f:
        runs = json.load(f)
    
    baseline_runs = [
        r for r in runs.values()
        if r.get("network_type") == "large_hidden"
        and r.get("input_type") == input_type
        and r.get("status") == "completed"
        and r.get("final_metrics") is not None
        and "final_win_rate" in r.get("final_metrics", {})
    ]
    
    if not baseline_runs:
        return None
    
    win_rates = [r["final_metrics"]["final_win_rate"] for r in baseline_runs]
    return sum(win_rates) / len(win_rates)


def get_baseline_win_rate(experiment_path: Path, runs: Dict, input_type: Optional[str] = None) -> Optional[float]:
    """Get average win rate for large_hidden baseline runs."""
    baseline_runs = [
        r for r in runs.values()
        if r.get("network_type") == "large_hidden"
        and (input_type is None or r.get("input_type") == input_type)
        and r.get("status") == "completed"
        and r.get("final_metrics") is not None
        and "final_win_rate" in r.get("final_metrics", {})
    ]
    
    if not baseline_runs:
        return None
    
    win_rates = [r["final_metrics"]["final_win_rate"] for r in baseline_runs]
    return sum(win_rates) / len(win_rates)


def get_scale_win_rate(runs: Dict, scale: int, input_type: Optional[str] = None) -> Optional[float]:
    """Get average win rate for a specific scale."""
    scale_runs = [
        r for r in runs.values()
        if r.get("scale") == scale
        and (input_type is None or r.get("input_type") == input_type)
        and r.get("status") == "completed"
        and r.get("final_metrics") is not None
        and "final_win_rate" in r.get("final_metrics", {})
    ]
    
    if not scale_runs:
        return None
    
    win_rates = [r["final_metrics"]["final_win_rate"] for r in scale_runs]
    return sum(win_rates) / len(win_rates)


def get_scale_completion_info(runs: Dict, scale: int) -> Dict[str, int]:
    """Get completion statistics for a specific scale."""
    scale_runs = [
        r for r in runs.values()
        if r.get("scale") == scale
    ]
    
    total = len(scale_runs)
    completed = sum(1 for r in scale_runs if r.get("status") == "completed")
    failed = sum(1 for r in scale_runs if r.get("status") == "failed")
    
    return {
        "total": total,
        "completed": completed,
        "failed": failed
    }


def run_experiment(
    experiment_path: Path,
    force: bool = False,
    networks: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
) -> None:
    """Run pending experiments, optionally filtered by network/input modes."""
    metadata, runs = load_experiment(experiment_path)
    
    # Check if this is a scaling experiment
    is_scaling_experiment = metadata.get("experiment_type") == "scaling"
    
    # Filter runs by status
    if force:
        run_ids = list(runs.keys())
    else:
        run_ids = [rid for rid, r in runs.items() if r["status"] == "pending"]
    
    # Filter by mode (network + input)
    if networks is not None or inputs is not None:
        allowed_networks = set(networks) if networks else set(NETWORK_TYPES)
        allowed_inputs = set(inputs) if inputs else set(INPUT_TYPES)
        run_ids = [
            rid
            for rid in run_ids
            if runs[rid]["network_type"] in allowed_networks
            and runs[rid]["input_type"] in allowed_inputs
        ]
        modes = [
            f"{n}_{i}"
            for n in (networks or NETWORK_TYPES)
            for i in (inputs or INPUT_TYPES)
        ]
        print(f"Modes: {', '.join(modes)}")
    
    if not run_ids:
        print("No runs to execute (all completed or none match criteria).")
        print_status(experiment_path)
        return
    
    # For scaling experiments, group by scale and run baseline first
    if is_scaling_experiment:
        # Separate baseline and scaling runs
        baseline_runs = [rid for rid in run_ids if runs[rid].get("network_type") == "large_hidden"]
        scaling_runs = [rid for rid in run_ids if runs[rid].get("network_type") == "game_based"]
        
        # Group scaling runs by scale
        runs_by_scale = {}
        for rid in scaling_runs:
            scale = runs[rid].get("scale")
            if scale is not None:
                if scale not in runs_by_scale:
                    runs_by_scale[scale] = []
                runs_by_scale[scale].append(rid)
        
        # Sort scales
        sorted_scales = sorted(runs_by_scale.keys())
        
        print(f"\nScaling experiment detected")
        print(f"Experiment: {experiment_path.name}")
        print(f"Total runs: {len(run_ids)}")
        print(f"  Baseline (large_hidden): {len(baseline_runs)} runs")
        print(f"  Scaling runs: {len(scaling_runs)} runs across {len(sorted_scales)} scales")
        print(f"\n{'='*70}")
        
        # Try to load baseline from another experiment first (per input type)
        baseline_win_rates = {}  # {input_type: win_rate}
        baseline_experiment_name = metadata.get("baseline_experiment")
        
        if baseline_experiment_name:
            baseline_experiment_path = experiments_dir / baseline_experiment_name
            if baseline_experiment_path.exists():
                print(f"\nLoading baseline win rates from: {baseline_experiment_name}")
                for input_type in ["boolean", "embedding"]:
                    win_rate = get_baseline_win_rate_from_experiment(baseline_experiment_path, input_type)
                    if win_rate is not None:
                        baseline_win_rates[input_type] = win_rate
                        print(f"  {input_type}: {win_rate:.1%}")
                if baseline_win_rates:
                    print(f"(Skipping baseline runs - using data from existing experiment)")
                else:
                    print(f"Warning: Could not load baseline from {baseline_experiment_name}")
                    print(f"Will run baseline runs instead")
        
        # Run baseline if not loaded from another experiment
        if not baseline_win_rates and baseline_runs:
            print(f"\nRunning baseline (large_hidden)...")
            for i, run_id in enumerate(baseline_runs, 1):
                run_info = runs[run_id]
                run_path = create_run_config(experiment_path, run_info)
                
                run_info["status"] = "running"
                run_info["created_at"] = datetime.now().isoformat()
                run_info["started_at"] = datetime.now().isoformat()
                save_runs(experiment_path, runs)
                
                success, metrics, error = run_single_training(experiment_path, run_info, run_path)
                
                run_info["status"] = "completed" if success else "failed"
                run_info["completed_at"] = datetime.now().isoformat()
                if run_info["started_at"]:
                    start = datetime.fromisoformat(run_info["started_at"])
                    end = datetime.now()
                    run_info["duration_seconds"] = (end - start).total_seconds()
                run_info["final_metrics"] = metrics
                run_info["error_message"] = error
                save_runs(experiment_path, runs)
                
                print(f"\nBaseline progress: {i}/{len(baseline_runs)} runs completed")
            
            # Reload runs to get updated baseline (per input type)
            metadata, runs = load_experiment(experiment_path)
            for input_type in ["boolean", "embedding"]:
                win_rate = get_baseline_win_rate(experiment_path, runs, input_type)
                if win_rate is not None:
                    baseline_win_rates[input_type] = win_rate
            
            if baseline_win_rates:
                print(f"\n{'='*70}")
                print(f"Baseline (large_hidden) average win rates:")
                for input_type, win_rate in baseline_win_rates.items():
                    print(f"  {input_type}: {win_rate:.1%}")
                print(f"{'='*70}")
        
        if not baseline_win_rates:
            print(f"\nWarning: No baseline win rate available. Will continue without baseline comparison.")
        
        # Group scaling runs by scale AND input type
        runs_by_scale_and_input = {}  # {(scale, input_type): [run_ids]}
        for rid in scaling_runs:
            scale = runs[rid].get("scale")
            input_type = runs[rid].get("input_type")
            if scale is not None and input_type is not None:
                key = (scale, input_type)
                if key not in runs_by_scale_and_input:
                    runs_by_scale_and_input[key] = []
                runs_by_scale_and_input[key].append(rid)
        
        # Run scaling runs by scale, then by input type
        total_completed = len(baseline_runs)
        paused = False
        
        for scale_idx, scale in enumerate(sorted_scales, 1):
            if paused:
                break
                
            hidden_layers = None
            for input_type in ["boolean", "embedding"]:
                key = (scale, input_type)
                if key not in runs_by_scale_and_input:
                    continue
                
                scale_run_ids = sorted(runs_by_scale_and_input[key])
                if hidden_layers is None:
                    hidden_layers = runs[scale_run_ids[0]].get("hidden_layers", [])
                
                print(f"\n{'='*70}")
                print(f"Running scale {scale} ({input_type}) (scale {scale_idx}/{len(sorted_scales)})")
                print(f"Hidden layers: {hidden_layers}")
                print(f"Runs: {len(scale_run_ids)}")
                print(f"{'='*70}")
                
                for i, run_id in enumerate(scale_run_ids, 1):
                    run_info = runs[run_id]
                    run_path = create_run_config(experiment_path, run_info)
                    
                    run_info["status"] = "running"
                    run_info["created_at"] = datetime.now().isoformat()
                    run_info["started_at"] = datetime.now().isoformat()
                    save_runs(experiment_path, runs)
                    
                    success, metrics, error = run_single_training(experiment_path, run_info, run_path)
                    
                    run_info["status"] = "completed" if success else "failed"
                    run_info["completed_at"] = datetime.now().isoformat()
                    if run_info["started_at"]:
                        start = datetime.fromisoformat(run_info["started_at"])
                        end = datetime.now()
                        run_info["duration_seconds"] = (end - start).total_seconds()
                    run_info["final_metrics"] = metrics
                    run_info["error_message"] = error
                    save_runs(experiment_path, runs)
                    
                    total_completed += 1
                    print(f"\nScale {scale} ({input_type}) progress: {i}/{len(scale_run_ids)} runs completed")
                    print(f"Overall progress: {total_completed}/{len(run_ids)} runs completed")
                
                # Check if this scale+input_type matches/exceeds baseline
                metadata, runs = load_experiment(experiment_path)  # Reload to get latest metrics
                baseline_win_rate = baseline_win_rates.get(input_type)
                scale_win_rate = get_scale_win_rate(runs, scale, input_type)
                completion_info = get_scale_completion_info(runs, scale)
                
                print(f"\n{'='*70}")
                print(f"Scale {scale} ({input_type}) completed")
                print(f"  Completion: {completion_info['completed']}/{completion_info['total']} runs completed")
                if completion_info['failed'] > 0:
                    print(f"  Failed: {completion_info['failed']} runs")
                
                if baseline_win_rate is not None and scale_win_rate is not None:
                    print(f"  Baseline (large_hidden, {input_type}): {baseline_win_rate:.1%}")
                    print(f"  Scale {scale} ({input_type}): {scale_win_rate:.1%}")
                    print(f"  Difference: {scale_win_rate - baseline_win_rate:+.1%}")
                    
                    if scale_win_rate >= baseline_win_rate:
                        print(f"\n{'*'*70}")
                        print(f"Scale {scale} ({input_type}) MATCHES or EXCEEDS baseline win rate!")
                        print(f"  Baseline ({input_type}): {baseline_win_rate:.1%}")
                        print(f"  Scale {scale} ({input_type}): {scale_win_rate:.1%}")
                        print(f"  Hidden layers: {hidden_layers}")
                        print(f"\nNote: Single run result - high variance. Review all results")
                        print(f"      to identify target scale for full experiment.")
                        print(f"{'*'*70}")
                        # Don't pause for screening - continue to see all scales
                elif baseline_win_rate is None:
                    print(f"  Warning: Baseline win rate ({input_type}) not available yet")
                elif scale_win_rate is None:
                    print(f"  Warning: Scale {scale} ({input_type}) win rate not available (runs may have failed)")
                print(f"{'='*70}")
        
        if not paused:
            print(f"\n{'='*70}")
            print("All scales completed!")
            print(f"{'='*70}")
            print_status(experiment_path)
        
        print(f"\n{'='*70}")
        print("All scales completed!")
        print(f"{'='*70}")
        print_status(experiment_path)
        return
    
    # Standard experiment: run sequentially
    print(f"\nExecuting {len(run_ids)} runs sequentially...")
    print(f"Experiment: {experiment_path.name}")
    
    for i, run_id in enumerate(run_ids, 1):
        run_info = runs[run_id]
        
        # Create run directory and config
        run_path = create_run_config(experiment_path, run_info)
        
        # Update status
        run_info["status"] = "running"
        run_info["created_at"] = datetime.now().isoformat()
        run_info["started_at"] = datetime.now().isoformat()
        save_runs(experiment_path, runs)
        
        # Execute training
        success, metrics, error = run_single_training(experiment_path, run_info, run_path)
        
        # Update status
        run_info["status"] = "completed" if success else "failed"
        run_info["completed_at"] = datetime.now().isoformat()
        if run_info["started_at"]:
            start = datetime.fromisoformat(run_info["started_at"])
            end = datetime.now()
            run_info["duration_seconds"] = (end - start).total_seconds()
        run_info["final_metrics"] = metrics
        run_info["error_message"] = error
        save_runs(experiment_path, runs)
        
        print(f"\nProgress: {i}/{len(run_ids)} runs completed")
    
    print(f"\n{'='*70}")
    print("Experiment completed!")
    print(f"{'='*70}")
    print_status(experiment_path)


def print_status(experiment_path: Path) -> None:
    """Print experiment status summary."""
    metadata, runs = load_experiment(experiment_path)
    
    print(f"\nExperiment: {metadata['display_name']}")
    print(f"Total runs: {len(runs)}")
    
    # Count by status
    status_counts = {}
    for run in runs.values():
        status = run["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nStatus:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Count by configuration
    print("\nBy configuration:")
    for network_type in NETWORK_TYPES:
        for input_type in INPUT_TYPES:
            config_runs = [r for r in runs.values() 
                          if r["network_type"] == network_type and r["input_type"] == input_type]
            completed = sum(1 for r in config_runs if r["status"] == "completed")
            total = len(config_runs)
            print(f"  {network_type} + {input_type}: {completed}/{total}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Experiment manager for architecture comparison")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new experiment")
    init_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    init_parser.add_argument("--description", "-d", default="", help="Experiment description")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run pending experiments")
    run_parser.add_argument("--force", "-f", action="store_true",
                           help="Re-run even if already completed")
    run_parser.add_argument(
        "--network", "-N",
        action="append",
        choices=NETWORK_TYPES,
        metavar="NETWORK",
        help="Run only these network types (default: all). May be repeated.",
    )
    run_parser.add_argument(
        "--input", "-I",
        action="append",
        choices=INPUT_TYPES,
        metavar="INPUT",
        help="Run only these input types (default: all). May be repeated.",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show experiment status")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_experiment(args.name, args.description)
    elif args.command == "run":
        experiment_path = get_current_experiment()
        if not experiment_path:
            print("No current experiment. Run 'experiment_manager.py init --name <name>' first.")
            sys.exit(1)
        run_experiment(
            experiment_path,
            force=args.force,
            networks=args.network,
            inputs=args.input,
        )
    elif args.command == "status":
        experiment_path = get_current_experiment()
        if not experiment_path:
            print("No current experiment found.")
            sys.exit(1)
        print_status(experiment_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
