#!/usr/bin/env python3
"""
Script to clear training logs from the action_logs directory.

This script removes all log files (actions_*.jsonl and metrics_*.jsonl)
from the action_logs directory.

Usage:
    python scripts/clear_logs.py                    # Interactive mode, all logs
    python scripts/clear_logs.py --force            # No confirmation, all logs
    python scripts/clear_logs.py --type hand_only    # Clear hand_only logs only
    python scripts/clear_logs.py --type opponent_field_only --force
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Valid training types
VALID_TRAINING_TYPES = [
    "hand_only",
    "opponent_field_only",
    "no_features",
    "both_features"
]


def get_log_files(log_dir: Path, training_type: Optional[str] = None) -> List[Path]:
    """
    Get all log files from the log directory and its subdirectories.
    
    Logs are now organized into subdirectories by training type:
    - action_logs/hand_only/
    - action_logs/opponent_field_only/
    - action_logs/no_features/
    - action_logs/both_features/
    
    Args:
        log_dir: Path to the log directory
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        List of log file paths
    """
    if not log_dir.exists():
        return []
    
    log_files = []
    
    if training_type:
        # Filter by specific training type subdirectory
        if training_type not in VALID_TRAINING_TYPES:
            print(f"Error: Invalid training type '{training_type}'")
            print(f"Valid types: {', '.join(VALID_TRAINING_TYPES)}")
            return []
        
        subdir = log_dir / training_type
        if subdir.exists():
            log_files.extend(subdir.glob("actions_*.jsonl"))
            log_files.extend(subdir.glob("metrics_*.jsonl"))
    else:
        # Find all log files recursively (including subdirectories)
        log_files.extend(log_dir.rglob("actions_*.jsonl"))
        log_files.extend(log_dir.rglob("metrics_*.jsonl"))
    
    return sorted(log_files)


def clear_logs(log_dir: Path, confirm: bool = True, training_type: Optional[str] = None) -> int:
    """
    Clear log files from the log directory.
    
    Args:
        log_dir: Path to the log directory
        confirm: If True, ask for confirmation before deleting
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of files deleted
    """
    log_files = get_log_files(log_dir, training_type)
    
    if not log_files:
        if training_type:
            print(f"No log files found for training type '{training_type}' in {log_dir}")
        else:
            print(f"No log files found in {log_dir}")
        return 0
    
    if training_type:
        print(f"Found {len(log_files)} log file(s) for training type '{training_type}':")
    else:
        print(f"Found {len(log_files)} log file(s):")
    for log_file in log_files:
        file_size = log_file.stat().st_size / (1024 * 1024)  # Size in MB
        # Show relative path from log_dir for better visibility
        rel_path = log_file.relative_to(log_dir)
        print(f"  - {rel_path} ({file_size:.2f} MB)")
    
    if confirm:
        response = input(f"\nDelete all {len(log_files)} log file(s)? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files deleted.")
            return 0
    
    # Delete files
    deleted_count = 0
    for log_file in log_files:
        try:
            rel_path = log_file.relative_to(log_dir)
            log_file.unlink()
            deleted_count += 1
            print(f"Deleted: {rel_path}")
        except Exception as e:
            rel_path = log_file.relative_to(log_dir)
            print(f"Error deleting {rel_path}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} log file(s)")
    return deleted_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clear training logs from the action_logs directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/clear_logs.py                    # Clear all logs (interactive)
  python scripts/clear_logs.py --force              # Clear all logs (no confirmation)
  python scripts/clear_logs.py --type hand_only    # Clear hand_only logs only
  python scripts/clear_logs.py --type opponent_field_only --force

Valid training types: {', '.join(VALID_TRAINING_TYPES)}
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=VALID_TRAINING_TYPES,
        help="Training type to filter by (e.g., hand_only, opponent_field_only)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Get the project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Log directory is in project root
    log_dir = project_root / "action_logs"
    
    if args.type:
        print(f"Log directory: {log_dir.absolute()}")
        print(f"Training type: {args.type}")
    else:
        print(f"Log directory: {log_dir.absolute()}")
        print("Training type: ALL")
    print()
    
    deleted = clear_logs(log_dir, confirm=not args.force, training_type=args.type)
    
    if deleted > 0:
        # Optionally remove empty subdirectories and the main directory if it's empty
        try:
            training_type = args.type
            if training_type:
                # Remove the specific subdirectory if it's empty
                subdir = log_dir / training_type
                if subdir.exists() and not any(subdir.iterdir()):
                    subdir.rmdir()
                    print(f"Removed empty subdirectory: {subdir.name}")
            else:
                # Remove all empty subdirectories
                for subdir in sorted(log_dir.iterdir(), reverse=True):
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                        print(f"Removed empty subdirectory: {subdir.name}")
            
            # Remove main directory if it's empty
            if log_dir.exists() and not any(log_dir.iterdir()):
                log_dir.rmdir()
                print(f"Removed empty directory: {log_dir}")
        except Exception as e:
            print(f"Note: Could not remove directory: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

