#!/usr/bin/env python3
"""
Script to clear training state files from the models directory.

This script removes training state files (training_state.json) that are used
to resume training from checkpoints.

Usage:
    python scripts/clear_training_state.py          # Interactive mode
    python scripts/clear_training_state.py --force  # No confirmation
"""

import sys
from pathlib import Path
from typing import List


def get_training_state_files(models_dir: Path) -> List[Path]:
    """
    Get all training state files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        List of training state file paths
    """
    if not models_dir.exists():
        return []
    
    training_state_files = []
    # Find training state JSON files
    training_state_files.extend(models_dir.glob("training_state.json"))
    # Also check for any other potential training state files
    training_state_files.extend(models_dir.glob("*training_state*"))
    
    # Remove duplicates and filter out directories
    training_state_files = [f for f in set(training_state_files) if f.is_file()]
    
    return sorted(training_state_files)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.23 KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def clear_training_state(models_dir: Path, confirm: bool = True) -> int:
    """
    Clear all training state files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        confirm: If True, ask for confirmation before deleting
        
    Returns:
        Number of files deleted
    """
    training_state_files = get_training_state_files(models_dir)
    
    if not training_state_files:
        print(f"No training state files found in {models_dir}")
        return 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in training_state_files)
    
    print(f"Found {len(training_state_files)} training state file(s):")
    for state_file in training_state_files:
        file_size = state_file.stat().st_size
        print(f"  - {state_file.name} ({format_file_size(file_size)})")
    
    print(f"\nTotal size: {format_file_size(total_size)}")
    
    if confirm:
        response = input(f"\nDelete all {len(training_state_files)} training state file(s)? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files deleted.")
            return 0
    
    # Delete files
    deleted_count = 0
    deleted_size = 0
    for state_file in training_state_files:
        try:
            file_size = state_file.stat().st_size
            state_file.unlink()
            deleted_count += 1
            deleted_size += file_size
            print(f"Deleted: {state_file.name} ({format_file_size(file_size)})")
        except Exception as e:
            print(f"Error deleting {state_file.name}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} training state file(s) ({format_file_size(deleted_size)})")
    return deleted_count


def main():
    """Main entry point for the script."""
    # Get the project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Models directory is in project root
    models_dir = project_root / "models"
    
    # Check for command line arguments
    force = "--force" in sys.argv or "-f" in sys.argv
    
    print(f"Models directory: {models_dir.absolute()}")
    print()
    
    deleted = clear_training_state(models_dir, confirm=not force)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

