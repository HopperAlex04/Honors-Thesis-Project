#!/usr/bin/env python3
"""
Script to clear training logs from the action_logs directory.

This script removes all log files (actions_*.jsonl and metrics_*.jsonl)
from the action_logs directory.

Usage:
    python scripts/clear_logs.py          # Interactive mode
    python scripts/clear_logs.py --force  # No confirmation
"""

import sys
from pathlib import Path
from typing import List


def get_log_files(log_dir: Path) -> List[Path]:
    """
    Get all log files from the log directory.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        List of log file paths
    """
    if not log_dir.exists():
        return []
    
    log_files = []
    # Find all action and metrics log files
    log_files.extend(log_dir.glob("actions_*.jsonl"))
    log_files.extend(log_dir.glob("metrics_*.jsonl"))
    
    return sorted(log_files)


def clear_logs(log_dir: Path, confirm: bool = True) -> int:
    """
    Clear all log files from the log directory.
    
    Args:
        log_dir: Path to the log directory
        confirm: If True, ask for confirmation before deleting
        
    Returns:
        Number of files deleted
    """
    log_files = get_log_files(log_dir)
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return 0
    
    print(f"Found {len(log_files)} log file(s):")
    for log_file in log_files:
        file_size = log_file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  - {log_file.name} ({file_size:.2f} MB)")
    
    if confirm:
        response = input(f"\nDelete all {len(log_files)} log file(s)? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files deleted.")
            return 0
    
    # Delete files
    deleted_count = 0
    for log_file in log_files:
        try:
            log_file.unlink()
            deleted_count += 1
            print(f"Deleted: {log_file.name}")
        except Exception as e:
            print(f"Error deleting {log_file.name}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} log file(s)")
    return deleted_count


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
    
    deleted = clear_logs(log_dir, confirm=not force)
    
    if deleted > 0:
        # Optionally remove the directory if it's empty
        try:
            if log_dir.exists() and not any(log_dir.iterdir()):
                log_dir.rmdir()
                print(f"Removed empty directory: {log_dir}")
        except Exception as e:
            print(f"Note: Could not remove directory: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

