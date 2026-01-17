#!/usr/bin/env python3
"""
Script to organize existing logs into subdirectories by training type.

This script moves log files from action_logs/ into subdirectories:
- action_logs/hand_only/
- action_logs/opponent_field_only/
- action_logs/no_features/
- action_logs/both_features/

Usage:
    python scripts/organize_logs.py          # Interactive mode
    python scripts/organize_logs.py --force  # No confirmation
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional


def extract_training_type(filename: str) -> Optional[str]:
    """
    Extract training type from log filename.
    
    Args:
        filename: Log filename (e.g., "actions_hand_only_round_0_selfplay.jsonl")
        
    Returns:
        Training type string (e.g., "hand_only") or None if not recognized
    """
    # Known training types
    training_types = [
        "hand_only",
        "opponent_field_only",
        "no_features",
        "both_features",  # Legacy name, kept for backward compatibility
        "all_features",
        "scores"
    ]
    
    # Check if filename contains any known training type
    for training_type in training_types:
        if training_type in filename:
            return training_type
    
    return None


def get_logs_to_organize(log_dir: Path) -> List[Tuple[Path, str]]:
    """
    Get all log files in the root directory that need to be organized.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        List of tuples (log_file_path, training_type)
    """
    if not log_dir.exists():
        return []
    
    logs_to_move = []
    
    # Find all log files in root directory only (not subdirectories)
    for log_file in log_dir.iterdir():
        if log_file.is_file() and log_file.suffix == ".jsonl":
            training_type = extract_training_type(log_file.name)
            if training_type:
                logs_to_move.append((log_file, training_type))
            else:
                print(f"Warning: Could not determine training type for {log_file.name}")
    
    return sorted(logs_to_move)


def organize_logs(log_dir: Path, confirm: bool = True) -> int:
    """
    Organize log files into subdirectories by training type.
    
    Args:
        log_dir: Path to the log directory
        confirm: If True, ask for confirmation before moving
        
    Returns:
        Number of files moved
    """
    logs_to_move = get_logs_to_organize(log_dir)
    
    if not logs_to_move:
        print(f"No log files found in {log_dir} that need organizing")
        return 0
    
    # Group by training type for display
    by_type = {}
    for log_file, training_type in logs_to_move:
        if training_type not in by_type:
            by_type[training_type] = []
        by_type[training_type].append(log_file)
    
    print(f"Found {len(logs_to_move)} log file(s) to organize:")
    print()
    for training_type in sorted(by_type.keys()):
        files = by_type[training_type]
        print(f"  {training_type}/: {len(files)} file(s)")
    
    if confirm:
        print()
        response = input(f"Move all {len(logs_to_move)} log file(s) into subdirectories? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files moved.")
            return 0
    
    # Move files
    moved_count = 0
    for log_file, training_type in logs_to_move:
        try:
            # Create subdirectory
            subdir = log_dir / training_type
            subdir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            dest_path = subdir / log_file.name
            log_file.rename(dest_path)
            moved_count += 1
            print(f"Moved: {log_file.name} -> {training_type}/{log_file.name}")
        except Exception as e:
            print(f"Error moving {log_file.name}: {e}")
    
    print(f"\n✓ Moved {moved_count} log file(s)")
    return moved_count


def main():
    """Main entry point for the script."""
    # Get the project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Log directory is in project root
    log_dir = project_root / "action_logs"
    
    # Check for command line arguments
    force = "--force" in sys.argv or "-f" in sys.argv
    
    print(f"Log directory: {log_dir.absolute()}")
    print()
    
    moved = organize_logs(log_dir, confirm=not force)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

