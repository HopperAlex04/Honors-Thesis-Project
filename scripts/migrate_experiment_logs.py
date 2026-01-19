#!/usr/bin/env python3
"""
Migrate Experiment Logs - Move metrics files from action_logs/ to metrics_logs/.

This script migrates existing experiments from the old structure (all logs in action_logs/)
to the new structure (action logs in action_logs/, metrics in metrics_logs/).

Usage:
    python scripts/migrate_experiment_logs.py                  # Dry-run (show what would be done)
    python scripts/migrate_experiment_logs.py --execute        # Actually move files
    python scripts/migrate_experiment_logs.py --experiment NAME  # Migrate specific experiment
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def find_experiments() -> List[Path]:
    """Find all experiment directories."""
    if not EXPERIMENTS_DIR.exists():
        return []
    
    return sorted([
        d for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("experiment_")
    ])


def find_runs(experiment_path: Path) -> List[Path]:
    """Find all run directories in an experiment."""
    runs_dir = experiment_path / "runs"
    if not runs_dir.exists():
        return []
    
    return sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and "_run_" in d.name
    ])


def get_metrics_files(action_logs_dir: Path) -> List[Path]:
    """Find all metrics files in action_logs directory."""
    if not action_logs_dir.exists():
        return []
    
    return sorted(action_logs_dir.glob("metrics_*.jsonl"))


def migrate_run(run_path: Path, dry_run: bool = True) -> Tuple[int, int]:
    """
    Migrate metrics files from action_logs/ to metrics_logs/ for a single run.
    
    Args:
        run_path: Path to the run directory
        dry_run: If True, only report what would be done without moving files
        
    Returns:
        Tuple of (files_moved, bytes_moved)
    """
    action_logs_dir = run_path / "action_logs"
    metrics_logs_dir = run_path / "metrics_logs"
    
    metrics_files = get_metrics_files(action_logs_dir)
    
    if not metrics_files:
        return 0, 0
    
    files_moved = 0
    bytes_moved = 0
    
    for metrics_file in metrics_files:
        file_size = metrics_file.stat().st_size
        dest_file = metrics_logs_dir / metrics_file.name
        
        if dry_run:
            print(f"  Would move: {metrics_file.name} ({file_size:,} bytes)")
        else:
            # Create metrics_logs directory if it doesn't exist
            metrics_logs_dir.mkdir(exist_ok=True)
            
            # Move the file
            shutil.move(str(metrics_file), str(dest_file))
            print(f"  Moved: {metrics_file.name} ({file_size:,} bytes)")
        
        files_moved += 1
        bytes_moved += file_size
    
    return files_moved, bytes_moved


def migrate_experiment(experiment_path: Path, dry_run: bool = True) -> Tuple[int, int, int]:
    """
    Migrate all runs in an experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        dry_run: If True, only report what would be done
        
    Returns:
        Tuple of (runs_processed, total_files, total_bytes)
    """
    runs = find_runs(experiment_path)
    
    if not runs:
        print(f"  No runs found in {experiment_path.name}")
        return 0, 0, 0
    
    total_files = 0
    total_bytes = 0
    runs_with_changes = 0
    
    for run_path in runs:
        files, bytes_moved = migrate_run(run_path, dry_run)
        if files > 0:
            runs_with_changes += 1
            total_files += files
            total_bytes += bytes_moved
    
    return runs_with_changes, total_files, total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Migrate metrics files from action_logs/ to metrics_logs/"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Migrate a specific experiment (by name or partial match)"
    )
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN - No files will be moved")
        print("Use --execute to actually move files")
        print("=" * 70)
        print()
    
    # Find experiments to migrate
    all_experiments = find_experiments()
    
    if not all_experiments:
        print("No experiments found in experiments/ directory")
        return 0
    
    # Filter by experiment name if specified
    if args.experiment:
        experiments = [e for e in all_experiments if args.experiment in e.name]
        if not experiments:
            print(f"No experiment matching '{args.experiment}' found")
            print(f"Available experiments: {[e.name for e in all_experiments]}")
            return 1
    else:
        experiments = all_experiments
    
    # Migrate each experiment
    grand_total_runs = 0
    grand_total_files = 0
    grand_total_bytes = 0
    
    for experiment_path in experiments:
        print(f"\nExperiment: {experiment_path.name}")
        print("-" * 50)
        
        runs, files, bytes_moved = migrate_experiment(experiment_path, dry_run)
        
        if files == 0:
            print("  No metrics files to migrate (already migrated or no logs)")
        else:
            print(f"  Summary: {runs} runs, {files} files, {bytes_moved:,} bytes")
        
        grand_total_runs += runs
        grand_total_files += files
        grand_total_bytes += bytes_moved
    
    # Print grand total
    print()
    print("=" * 70)
    if dry_run:
        print(f"WOULD MIGRATE: {grand_total_files} files ({grand_total_bytes:,} bytes) across {grand_total_runs} runs")
        print()
        print("Run with --execute to perform the migration")
    else:
        print(f"MIGRATED: {grand_total_files} files ({grand_total_bytes:,} bytes) across {grand_total_runs} runs")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
