#!/usr/bin/env python3
"""
Script to clear training state files from the models directory.

This script removes training state files (training_state_*.json) that are used
to resume training from checkpoints.

Usage:
    python scripts/clear_training_state.py                    # Interactive mode, all states
    python scripts/clear_training_state.py --force              # No confirmation, all states
    python scripts/clear_training_state.py --type hand_only    # Clear hand_only state only
    python scripts/clear_training_state.py --type opponent_field_only --force
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
    "both_features",  # Legacy name, kept for backward compatibility
    "all_features",
    "scores"
]


def get_training_state_files(models_dir: Path, training_type: Optional[str] = None) -> List[Path]:
    """
    Get all training state files from the models directory.
    
    Training state files are named like:
    - training_state_hand_only.json
    - training_state_opponent_field_only.json
    - training_state_no_features.json
    - training_state_both_features.json (legacy)
    - training_state_all_features.json
    - training_state_scores.json
    
    Args:
        models_dir: Path to the models directory
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        List of training state file paths
    """
    if not models_dir.exists():
        return []
    
    training_state_files = []
    
    if training_type:
        # Filter by specific training type
        if training_type not in VALID_TRAINING_TYPES:
            print(f"Error: Invalid training type '{training_type}'")
            print(f"Valid types: {', '.join(VALID_TRAINING_TYPES)}")
            return []
        
        # Find training state files with the training type suffix
        training_state_files.extend(models_dir.glob(f"training_state_{training_type}.json"))
    else:
        # Find all training state JSON files
        training_state_files.extend(models_dir.glob("training_state*.json"))
        # Also check for generic training_state.json (legacy)
        if (models_dir / "training_state.json").exists():
            training_state_files.append(models_dir / "training_state.json")
    
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


def clear_training_state(models_dir: Path, confirm: bool = True, training_type: Optional[str] = None) -> int:
    """
    Clear training state files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        confirm: If True, ask for confirmation before deleting
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of files deleted
    """
    training_state_files = get_training_state_files(models_dir, training_type)
    
    if not training_state_files:
        if training_type:
            print(f"No training state files found for training type '{training_type}' in {models_dir}")
        else:
            print(f"No training state files found in {models_dir}")
        return 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in training_state_files)
    
    if training_type:
        print(f"Found {len(training_state_files)} training state file(s) for training type '{training_type}':")
    else:
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
    parser = argparse.ArgumentParser(
        description="Clear training state files from the models directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/clear_training_state.py                    # Clear all states (interactive)
  python scripts/clear_training_state.py --force              # Clear all states (no confirmation)
  python scripts/clear_training_state.py --type hand_only    # Clear hand_only state only
  python scripts/clear_training_state.py --type opponent_field_only --force

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
    
    # Models directory is in project root
    models_dir = project_root / "models"
    
    if args.type:
        print(f"Models directory: {models_dir.absolute()}")
        print(f"Training type: {args.type}")
    else:
        print(f"Models directory: {models_dir.absolute()}")
        print("Training type: ALL")
    print()
    
    deleted = clear_training_state(models_dir, confirm=not args.force, training_type=args.type)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

