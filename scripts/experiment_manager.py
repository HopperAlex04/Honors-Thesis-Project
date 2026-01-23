#!/usr/bin/env python3
"""
Experiment Manager for Architecture Comparison Experiments.

This script manages the full experimental setup for comparing 3 network architectures
(Linear, Large Hidden, Game-Based) with 7 runs each (21 total runs).

Features:
- Creates organized experiment directory structure
- Manages individual run configurations with unique seeds
- Tracks run status and progress
- Aggregates results across runs
- Performs statistical analysis
- Generates comparison visualizations

Usage:
    python scripts/experiment_manager.py init --name "architecture_comparison_v1"
    python scripts/experiment_manager.py status
    python scripts/experiment_manager.py run --type linear --run 1
    python scripts/experiment_manager.py run-all --parallel 3
    python scripts/experiment_manager.py analyze
    python scripts/experiment_manager.py export --format csv
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Constants
NETWORK_TYPES = ["linear", "large_hidden", "game_based"]
RUNS_PER_TYPE = 5  # Reduced from 7 to 5 for time efficiency while maintaining statistical significance
TOTAL_RUNS = len(NETWORK_TYPES) * RUNS_PER_TYPE

EXPERIMENTS_DIR = project_root / "experiments"
CURRENT_EXPERIMENT_FILE = EXPERIMENTS_DIR / ".current_experiment"


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


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def generate_seeds(count: int, base_seed: int = 42) -> List[int]:
    """Generate deterministic random seeds for reproducibility."""
    np.random.seed(base_seed)
    return [int(np.random.randint(1, 100000)) for _ in range(count)]


class ExperimentManager:
    """Manages full experiment lifecycle for input representation comparison."""

    def __init__(self, experiment_path: Optional[Path] = None):
        """
        Initialize the experiment manager.
        
        Args:
            experiment_path: Path to experiment directory. If None, uses current experiment.
        """
        self.experiment_path = experiment_path
        self.metadata: Dict[str, Any] = {}
        self.runs: Dict[str, Dict[str, Any]] = {}
        
        if experiment_path and experiment_path.exists():
            self._load_experiment()

    def _load_experiment(self) -> None:
        """Load experiment metadata and run status."""
        metadata_file = self.experiment_path / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        runs_file = self.experiment_path / "runs_status.json"
        if runs_file.exists():
            with open(runs_file, 'r') as f:
                self.runs = json.load(f)

    def _save_experiment(self) -> None:
        """Save experiment metadata and run status."""
        metadata_file = self.experiment_path / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        runs_file = self.experiment_path / "runs_status.json"
        with open(runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2)

    @staticmethod
    def init_experiment(name: str, description: str = "", base_config_path: Optional[Path] = None) -> 'ExperimentManager':
        """
        Initialize a new experiment.
        
        Args:
            name: Experiment name (used in directory name)
            description: Description of the experiment
            base_config_path: Path to base hyperparameters config file
            
        Returns:
            Initialized ExperimentManager instance
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}_{name}"
        experiment_path = EXPERIMENTS_DIR / experiment_name
        
        # Create directory structure
        experiment_path.mkdir(parents=True, exist_ok=True)
        (experiment_path / "runs").mkdir(exist_ok=True)
        (experiment_path / "analysis").mkdir(exist_ok=True)
        (experiment_path / "analysis" / "graphs").mkdir(exist_ok=True)
        
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
            "runs_per_type": RUNS_PER_TYPE,
            "total_runs": TOTAL_RUNS,
            "status": "initialized",
            "seeds": seeds,
        }
        
        # Copy base config if provided
        if base_config_path and base_config_path.exists():
            shutil.copy(base_config_path, experiment_path / "base_hyperparams_config.json")
            metadata["base_config"] = "base_hyperparams_config.json"
        else:
            # Copy current config
            current_config = project_root / "hyperparams_config.json"
            if current_config.exists():
                shutil.copy(current_config, experiment_path / "base_hyperparams_config.json")
                metadata["base_config"] = "base_hyperparams_config.json"
        
        # Initialize run tracking
        runs = {}
        run_idx = 0
        for network_type in NETWORK_TYPES:
            for run_num in range(1, RUNS_PER_TYPE + 1):
                run_id = f"{network_type}_run_{run_num:02d}"
                runs[run_id] = {
                    "run_id": run_id,
                    "network_type": network_type,
                    "run_number": run_num,
                    "seed": seeds[run_idx],
                    "status": "pending",
                    "created_at": None,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "final_metrics": None,
                    "error_message": None,
                }
                run_idx += 1
        
        # Save everything
        manager = ExperimentManager(experiment_path)
        manager.metadata = metadata
        manager.runs = runs
        manager._save_experiment()
        
        # Update current experiment pointer
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_EXPERIMENT_FILE, 'w') as f:
            f.write(str(experiment_path))
        
        print(f"Initialized experiment: {experiment_name}")
        print(f"Location: {experiment_path}")
        print(f"Network types: {', '.join(NETWORK_TYPES)}")
        print(f"Runs per type: {RUNS_PER_TYPE}")
        print(f"Total runs: {TOTAL_RUNS}")
        
        return manager

    @staticmethod
    def get_current() -> Optional['ExperimentManager']:
        """Get the current active experiment."""
        if not CURRENT_EXPERIMENT_FILE.exists():
            return None
        
        with open(CURRENT_EXPERIMENT_FILE, 'r') as f:
            experiment_path = Path(f.read().strip())
        
        if not experiment_path.exists():
            return None
        
        return ExperimentManager(experiment_path)

    def get_run_path(self, run_id: str) -> Path:
        """Get the path for a specific run's directory."""
        return self.experiment_path / "runs" / run_id

    def prepare_run(self, run_id: str) -> Path:
        """
        Prepare a run directory with its configuration.
        
        Args:
            run_id: Run identifier (e.g., "boolean_run_01")
            
        Returns:
            Path to the run directory
        """
        if run_id not in self.runs:
            raise ValueError(f"Unknown run ID: {run_id}")
        
        run_info = self.runs[run_id]
        run_path = self.get_run_path(run_id)
        
        # Create run directory structure
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "models").mkdir(exist_ok=True)
        (run_path / "action_logs").mkdir(exist_ok=True)
        (run_path / "metrics_logs").mkdir(exist_ok=True)
        
        # Create run-specific config
        base_config_path = self.experiment_path / "base_hyperparams_config.json"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update config for this run
        config["network_type"] = run_info["network_type"]
        config["random_seed"] = run_info["seed"]
        
        # Disable quick test mode for actual experiments
        if "training" in config:
            config["training"]["quick_test_mode"] = False
        
        # Save run config
        config_path = run_path / "hyperparams_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create run metadata
        run_metadata = {
            "run_id": run_id,
            "network_type": run_info["network_type"],
            "run_number": run_info["run_number"],
            "seed": run_info["seed"],
            "experiment_name": self.metadata["experiment_name"],
            "created_at": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
        }
        with open(run_path / "run_metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        # Update run status
        run_info["created_at"] = datetime.now().isoformat()
        run_info["status"] = "prepared"
        self._save_experiment()
        
        return run_path

    def start_run(self, run_id: str) -> None:
        """Mark a run as started."""
        if run_id not in self.runs:
            raise ValueError(f"Unknown run ID: {run_id}")
        
        self.runs[run_id]["started_at"] = datetime.now().isoformat()
        self.runs[run_id]["status"] = "running"
        self._save_experiment()

    def complete_run(self, run_id: str, metrics: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """
        Mark a run as completed.
        
        Args:
            run_id: Run identifier
            metrics: Final metrics from the run
            error: Error message if run failed
        """
        if run_id not in self.runs:
            raise ValueError(f"Unknown run ID: {run_id}")
        
        run_info = self.runs[run_id]
        run_info["completed_at"] = datetime.now().isoformat()
        
        if error:
            run_info["status"] = "failed"
            run_info["error_message"] = error
        else:
            run_info["status"] = "completed"
            run_info["final_metrics"] = metrics
        
        # Calculate duration if we have start time
        if run_info["started_at"]:
            start = datetime.fromisoformat(run_info["started_at"])
            end = datetime.fromisoformat(run_info["completed_at"])
            run_info["duration_seconds"] = (end - start).total_seconds()
        
        self._save_experiment()

    def get_status(self) -> Dict[str, Any]:
        """Get overall experiment status."""
        status_counts = {"pending": 0, "prepared": 0, "running": 0, "completed": 0, "failed": 0}
        for run_info in self.runs.values():
            status_counts[run_info["status"]] += 1
        
        completed_runs = [r for r in self.runs.values() if r["status"] == "completed"]
        total_duration = sum(r.get("duration_seconds", 0) or 0 for r in completed_runs)
        
        return {
            "experiment_name": self.metadata["experiment_name"],
            "status": self.metadata["status"],
            "total_runs": TOTAL_RUNS,
            "by_status": status_counts,
            "completed_duration_total": total_duration,
            "average_run_duration": total_duration / len(completed_runs) if completed_runs else 0,
        }

    def get_runs_by_type(self, network_type: str) -> List[Dict[str, Any]]:
        """Get all runs for a specific network type."""
        return [r for r in self.runs.values() if r["network_type"] == network_type]

    def get_pending_runs(self) -> List[str]:
        """Get list of run IDs that haven't been completed."""
        return [
            run_id for run_id, info in self.runs.items()
            if info["status"] in ["pending", "prepared"]
        ]

    def print_status(self) -> None:
        """Print detailed status of the experiment."""
        status = self.get_status()
        
        print(f"\n{'='*70}")
        print(f"Experiment: {self.metadata.get('display_name', self.metadata['experiment_name'])}")
        print(f"{'='*70}")
        
        if self.metadata.get("description"):
            print(f"Description: {self.metadata['description']}")
        
        print(f"\nCreated: {self.metadata.get('created_at', 'Unknown')}")
        print(f"Git commit: {self.metadata.get('git_commit', 'Unknown')}")
        
        print(f"\nProgress: {status['by_status']['completed']}/{status['total_runs']} runs completed")
        print(f"  - Pending: {status['by_status']['pending']}")
        print(f"  - Prepared: {status['by_status']['prepared']}")
        print(f"  - Running: {status['by_status']['running']}")
        print(f"  - Completed: {status['by_status']['completed']}")
        print(f"  - Failed: {status['by_status']['failed']}")
        
        if status['average_run_duration'] > 0:
            print(f"\nAverage run duration: {format_duration(status['average_run_duration'])}")
            remaining = status['by_status']['pending'] + status['by_status']['prepared']
            if remaining > 0:
                estimated = remaining * status['average_run_duration']
                print(f"Estimated time remaining (sequential): {format_duration(estimated)}")
        
        # Print per-type breakdown
        print(f"\n{'='*70}")
        print("Status by Network Type:")
        print(f"{'='*70}")
        
        for network_type in NETWORK_TYPES:
            runs = self.get_runs_by_type(network_type)
            completed = sum(1 for r in runs if r["status"] == "completed")
            failed = sum(1 for r in runs if r["status"] == "failed")
            
            status_str = f"{completed}/{len(runs)} completed"
            if failed > 0:
                status_str += f" ({failed} failed)"
            
            print(f"  {network_type:15s}: {status_str}")
            
            # Show completed runs with metrics
            completed_runs = [r for r in runs if r["status"] == "completed" and r.get("final_metrics")]
            if completed_runs:
                win_rates = [r["final_metrics"].get("final_win_rate", 0) for r in completed_runs]
                if win_rates:
                    print(f"    Win rates: {[f'{wr:.1%}' for wr in win_rates]}")
        
        print()


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new experiment."""
    base_config = Path(args.config) if args.config else None
    ExperimentManager.init_experiment(
        name=args.name,
        description=args.description or "",
        base_config_path=base_config
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show experiment status."""
    manager = ExperimentManager.get_current()
    if not manager:
        print("No current experiment. Run 'init' first.")
        return 1
    
    manager.print_status()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all experiments."""
    if not EXPERIMENTS_DIR.exists():
        print("No experiments directory found.")
        return 0
    
    experiments = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("experiment_")])
    
    if not experiments:
        print("No experiments found.")
        return 0
    
    # Get current experiment
    current = None
    if CURRENT_EXPERIMENT_FILE.exists():
        with open(CURRENT_EXPERIMENT_FILE, 'r') as f:
            current = Path(f.read().strip())
    
    print(f"\n{'='*70}")
    print("Experiments:")
    print(f"{'='*70}")
    
    for exp_path in experiments:
        is_current = " (current)" if exp_path == current else ""
        metadata_file = exp_path / "experiment_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            runs_file = exp_path / "runs_status.json"
            completed = 0
            if runs_file.exists():
                with open(runs_file, 'r') as f:
                    runs = json.load(f)
                completed = sum(1 for r in runs.values() if r["status"] == "completed")
            
            print(f"\n  {metadata.get('display_name', exp_path.name)}{is_current}")
            print(f"    Path: {exp_path.name}")
            print(f"    Created: {metadata.get('created_at', 'Unknown')[:19]}")
            print(f"    Progress: {completed}/{metadata.get('total_runs', '?')} runs")
        else:
            print(f"\n  {exp_path.name}{is_current}")
            print(f"    (metadata missing)")
    
    print()
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    """Select an experiment as current."""
    # Find matching experiment
    if not EXPERIMENTS_DIR.exists():
        print("No experiments directory found.")
        return 1
    
    matches = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and args.name in d.name]
    
    if not matches:
        print(f"No experiment matching '{args.name}' found.")
        return 1
    
    if len(matches) > 1:
        print(f"Multiple matches found:")
        for m in matches:
            print(f"  - {m.name}")
        print("Please be more specific.")
        return 1
    
    experiment_path = matches[0]
    with open(CURRENT_EXPERIMENT_FILE, 'w') as f:
        f.write(str(experiment_path))
    
    print(f"Selected experiment: {experiment_path.name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a single run."""
    manager = ExperimentManager.get_current()
    if not manager:
        print("No current experiment. Run 'init' first.")
        return 1
    
    run_id = f"{args.type}_run_{args.run:02d}"
    if run_id not in manager.runs:
        print(f"Invalid run ID: {run_id}")
        print(f"Valid types: {NETWORK_TYPES}")
        print(f"Valid run numbers: 1-{RUNS_PER_TYPE}")
        return 1
    
    run_info = manager.runs[run_id]
    if run_info["status"] == "completed" and not args.force:
        print(f"Run {run_id} is already completed. Use --force to re-run.")
        return 1
    
    print(f"\n{'='*70}")
    print(f"Starting run: {run_id}")
    print(f"Network type: {args.type}")
    print(f"Seed: {run_info['seed']}")
    print(f"{'='*70}\n")
    
    # Prepare run directory
    run_path = manager.prepare_run(run_id)
    print(f"Run directory: {run_path}")
    
    # Update status
    manager.start_run(run_id)
    
    # Execute training
    try:
        # Set up environment for the run
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "src")
        
        # Run training with the run-specific config
        train_script = project_root / "train.py"
        config_file = run_path / "hyperparams_config.json"
        
        # Temporarily copy config to project root (train.py expects it there)
        original_config = project_root / "hyperparams_config.json"
        backup_config = project_root / "hyperparams_config.json.bak"
        
        if original_config.exists():
            shutil.copy(original_config, backup_config)
        
        shutil.copy(config_file, original_config)
        
        # Create models, action_logs, and metrics_logs symlinks/directories for this run
        models_dir = project_root / "models"
        action_logs_dir = project_root / "action_logs"
        metrics_logs_dir = project_root / "metrics_logs"
        
        # Backup existing directories
        if models_dir.exists() and not models_dir.is_symlink():
            shutil.move(models_dir, project_root / "models.bak")
        if action_logs_dir.exists() and not action_logs_dir.is_symlink():
            shutil.move(action_logs_dir, project_root / "action_logs.bak")
        if metrics_logs_dir.exists() and not metrics_logs_dir.is_symlink():
            shutil.move(metrics_logs_dir, project_root / "metrics_logs.bak")
        
        # Create symlinks to run directories
        if models_dir.is_symlink():
            models_dir.unlink()
        if action_logs_dir.is_symlink():
            action_logs_dir.unlink()
        if metrics_logs_dir.is_symlink():
            metrics_logs_dir.unlink()
        
        models_dir.symlink_to(run_path / "models")
        action_logs_dir.symlink_to(run_path / "action_logs")
        metrics_logs_dir.symlink_to(run_path / "metrics_logs")
        
        try:
            # Run training
            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=project_root,
                env=env,
                text=True,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Training failed with return code {result.returncode}")
            
            # Collect final metrics
            metrics = collect_run_metrics(run_path)
            manager.complete_run(run_id, metrics=metrics)
            
            print(f"\n{'='*70}")
            print(f"Run {run_id} completed successfully!")
            if metrics:
                print(f"Final win rate: {metrics.get('final_win_rate', 'N/A'):.1%}")
            print(f"{'='*70}\n")
            
        finally:
            # Restore original config
            if backup_config.exists():
                shutil.move(backup_config, original_config)
            
            # Remove symlinks and restore backups
            if models_dir.is_symlink():
                models_dir.unlink()
            if action_logs_dir.is_symlink():
                action_logs_dir.unlink()
            if metrics_logs_dir.is_symlink():
                metrics_logs_dir.unlink()
            
            if (project_root / "models.bak").exists():
                shutil.move(project_root / "models.bak", models_dir)
            if (project_root / "action_logs.bak").exists():
                shutil.move(project_root / "action_logs.bak", action_logs_dir)
            if (project_root / "metrics_logs.bak").exists():
                shutil.move(project_root / "metrics_logs.bak", metrics_logs_dir)
        
        return 0
        
    except Exception as e:
        manager.complete_run(run_id, error=str(e))
        print(f"\nRun {run_id} failed: {e}")
        return 1


def collect_run_metrics(run_path: Path) -> Dict[str, Any]:
    """Collect final metrics from a completed run."""
    metrics = {}
    
    # Find metrics files (check metrics_logs first, fall back to action_logs for legacy)
    metrics_logs = run_path / "metrics_logs"
    action_logs = run_path / "action_logs"
    
    log_dir = metrics_logs if metrics_logs.exists() else action_logs
    if not log_dir.exists():
        return metrics
    
    # Look for validation metrics (vs_randomized or vs_gapmaximizer)
    # Files are directly in the log directory, not in subdirectories
    validation_files = sorted(log_dir.glob("metrics_*_vs_*.jsonl"))
    if validation_files:
        last_file = validation_files[-1]
        
        # Read last line to get final metrics
        with open(last_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_metrics = json.loads(lines[-1])
                metrics["final_win_rate"] = last_metrics.get("p1_win_rate", 0)
                metrics["final_episode"] = last_metrics.get("episode", 0)
    
    # Get metrics from final model if available
    models_dir = run_path / "models"
    final_model = models_dir / "model_final.pt"
    if final_model.exists():
        try:
            import torch
            checkpoint = torch.load(final_model, weights_only=False)
            if isinstance(checkpoint, dict):
                metrics["total_time"] = checkpoint.get("total_time", 0)
                metrics["steps_done"] = checkpoint.get("steps_done", 0)
                if "win_rate_history" in checkpoint:
                    # Get final win rate from history
                    for opponent, history in checkpoint["win_rate_history"].items():
                        if history:
                            metrics["final_win_rate"] = history[-1]
                            break
        except Exception:
            pass  # If we can't load the model, just use metrics from logs
    
    return metrics


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze experiment results and generate statistics."""
    manager = ExperimentManager.get_current()
    if not manager:
        print("No current experiment. Run 'init' first.")
        return 1
    
    print(f"\n{'='*70}")
    print("Experiment Analysis")
    print(f"{'='*70}\n")
    
    # Collect metrics by network type
    results_by_type: Dict[str, List[float]] = {t: [] for t in NETWORK_TYPES}
    
    for run_id, run_info in manager.runs.items():
        if run_info["status"] == "completed" and run_info.get("final_metrics"):
            win_rate = run_info["final_metrics"].get("final_win_rate", 0)
            results_by_type[run_info["network_type"]].append(win_rate)
    
    # Calculate statistics
    stats = {}
    for network_type, win_rates in results_by_type.items():
        if win_rates:
            stats[network_type] = {
                "n": len(win_rates),
                "mean": np.mean(win_rates),
                "std": np.std(win_rates, ddof=1) if len(win_rates) > 1 else 0,
                "min": np.min(win_rates),
                "max": np.max(win_rates),
                "median": np.median(win_rates),
                "values": win_rates,
            }
            
            # Calculate 95% confidence interval
            if len(win_rates) >= 2:
                from scipy import stats as scipy_stats
                ci = scipy_stats.t.interval(
                    0.95,
                    len(win_rates) - 1,
                    loc=np.mean(win_rates),
                    scale=scipy_stats.sem(win_rates)
                )
                stats[network_type]["ci_95_lower"] = ci[0]
                stats[network_type]["ci_95_upper"] = ci[1]
    
    # Print results
    print("Win Rate Statistics by Network Type:")
    print("-" * 70)
    print(f"{'Type':<15} {'N':>3} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'Range':>15}")
    print("-" * 70)
    
    for network_type in NETWORK_TYPES:
        if network_type in stats:
            s = stats[network_type]
            ci_str = f"[{s.get('ci_95_lower', 0):.1%}, {s.get('ci_95_upper', 0):.1%}]" if 'ci_95_lower' in s else "N/A"
            range_str = f"[{s['min']:.1%}, {s['max']:.1%}]"
            print(f"{network_type:<15} {s['n']:>3} {s['mean']:>7.1%} {s['std']:>7.1%} {ci_str:>20} {range_str:>15}")
        else:
            print(f"{network_type:<15} {'No completed runs':>50}")
    
    print()
    
    # Perform statistical tests if we have enough data
    if all(len(results_by_type[t]) >= 2 for t in NETWORK_TYPES):
        print("Statistical Comparisons (t-tests):")
        print("-" * 70)
        
        try:
            from scipy import stats as scipy_stats
            
            comparisons = [
                ("linear", "large_hidden"),
                ("linear", "game_based"),
                ("large_hidden", "game_based"),
            ]
            
            for t1, t2 in comparisons:
                t_stat, p_value = scipy_stats.ttest_ind(
                    results_by_type[t1],
                    results_by_type[t2]
                )
                sig = "*" if p_value < 0.05 else ""
                sig += "*" if p_value < 0.01 else ""
                sig += "*" if p_value < 0.001 else ""
                
                print(f"  {t1} vs {t2}: t={t_stat:.3f}, p={p_value:.4f} {sig}")
            
            print("\n  * p < 0.05, ** p < 0.01, *** p < 0.001")
            
        except ImportError:
            print("  (scipy not available for statistical tests)")
    
    # Save analysis results
    analysis_file = manager.experiment_path / "analysis" / "statistics.json"
    analysis_results = {
        "generated_at": datetime.now().isoformat(),
        "statistics": {k: {**v, "values": [float(x) for x in v["values"]]} for k, v in stats.items()},
    }
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis saved to: {analysis_file}")
    
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export experiment data."""
    manager = ExperimentManager.get_current()
    if not manager:
        print("No current experiment. Run 'init' first.")
        return 1
    
    export_path = manager.experiment_path / "analysis" / f"export.{args.format}"
    
    # Collect all run data
    rows = []
    for run_id, run_info in manager.runs.items():
        row = {
            "run_id": run_id,
            "network_type": run_info["network_type"],
            "run_number": run_info["run_number"],
            "seed": run_info["seed"],
            "status": run_info["status"],
            "duration_seconds": run_info.get("duration_seconds"),
        }
        
        if run_info.get("final_metrics"):
            for k, v in run_info["final_metrics"].items():
                row[f"metric_{k}"] = v
        
        rows.append(row)
    
    if args.format == "csv":
        import csv
        
        if rows:
            fieldnames = rows[0].keys()
            with open(export_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    elif args.format == "json":
        with open(export_path, 'w') as f:
            json.dump(rows, f, indent=2)
    
    print(f"Exported to: {export_path}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment Manager for Input Representation Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new experiment")
    init_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    init_parser.add_argument("--description", "-d", help="Experiment description")
    init_parser.add_argument("--config", "-c", help="Path to base config file")
    
    # status command
    subparsers.add_parser("status", help="Show current experiment status")
    
    # list command
    subparsers.add_parser("list", help="List all experiments")
    
    # select command
    select_parser = subparsers.add_parser("select", help="Select an experiment as current")
    select_parser.add_argument("name", help="Experiment name (partial match)")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Execute a single run")
    run_parser.add_argument("--type", "-t", required=True, choices=NETWORK_TYPES, help="Network type")
    run_parser.add_argument("--run", "-r", type=int, required=True, help=f"Run number (1-{RUNS_PER_TYPE})")
    run_parser.add_argument("--force", "-f", action="store_true", help="Force re-run if already completed")
    
    # analyze command
    subparsers.add_parser("analyze", help="Analyze experiment results")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export experiment data")
    export_parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", help="Export format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "init": cmd_init,
        "status": cmd_status,
        "list": cmd_list,
        "select": cmd_select,
        "run": cmd_run,
        "analyze": cmd_analyze,
        "export": cmd_export,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
