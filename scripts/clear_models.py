#!/usr/bin/env python3
"""
Script to clear model checkpoints from the models directory.

This script removes all model files (.pt, .pth) and checkpoint files
from the models directory.

Usage:
    python scripts/clear_models.py                    # Interactive mode, all models
    python scripts/clear_models.py --force          # No confirmation, all models
    python scripts/clear_models.py --type hand_only # Clear hand_only models only
    python scripts/clear_models.py --type opponent_field_only --force
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


def get_model_files(models_dir: Path, training_type: Optional[str] = None) -> List[Path]:
    """
    Get all model files from the models directory.
    
    Model files are named like:
    - hand_only_checkpoint0.pt
    - opponent_field_only_checkpoint0.pt
    - no_features_checkpoint0.pt
    - both_features_checkpoint0.pt
    
    Args:
        models_dir: Path to the models directory
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        List of model file paths
    """
    if not models_dir.exists():
        return []
    
    model_files = []
    
    if training_type:
        # Filter by specific training type prefix
        if training_type not in VALID_TRAINING_TYPES:
            print(f"Error: Invalid training type '{training_type}'")
            print(f"Valid types: {', '.join(VALID_TRAINING_TYPES)}")
            return []
        
        # Find model files with the training type prefix
        model_files.extend(models_dir.glob(f"{training_type}_*.pt"))
        model_files.extend(models_dir.glob(f"{training_type}_*.pth"))
        model_files.extend(models_dir.glob(f"{training_type}_checkpoint*"))
    else:
        # Find all PyTorch model files
        model_files.extend(models_dir.glob("*.pt"))
        model_files.extend(models_dir.glob("*.pth"))
        # Also check for checkpoint files without extension (if they exist)
        model_files.extend(models_dir.glob("checkpoint*"))
    
    # Remove duplicates and filter out directories
    model_files = [f for f in set(model_files) if f.is_file()]
    
    return sorted(model_files)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def clear_models(models_dir: Path, confirm: bool = True, training_type: Optional[str] = None) -> int:
    """
    Clear model files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        confirm: If True, ask for confirmation before deleting
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of files deleted
    """
    model_files = get_model_files(models_dir, training_type)
    
    if not model_files:
        if training_type:
            print(f"No model files found for training type '{training_type}' in {models_dir}")
        else:
            print(f"No model files found in {models_dir}")
        return 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_files)
    
    if training_type:
        print(f"Found {len(model_files)} model file(s) for training type '{training_type}':")
    else:
        print(f"Found {len(model_files)} model file(s):")
    for model_file in model_files:
        file_size = model_file.stat().st_size
        print(f"  - {model_file.name} ({format_file_size(file_size)})")
    
    print(f"\nTotal size: {format_file_size(total_size)}")
    
    if confirm:
        response = input(f"\nDelete all {len(model_files)} model file(s)? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files deleted.")
            return 0
    
    # Delete files
    deleted_count = 0
    deleted_size = 0
    for model_file in model_files:
        try:
            file_size = model_file.stat().st_size
            model_file.unlink()
            deleted_count += 1
            deleted_size += file_size
            print(f"Deleted: {model_file.name} ({format_file_size(file_size)})")
        except Exception as e:
            print(f"Error deleting {model_file.name}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} model file(s) ({format_file_size(deleted_size)})")
    return deleted_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clear model checkpoints from the models directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/clear_models.py                    # Clear all models (interactive)
  python scripts/clear_models.py --force              # Clear all models (no confirmation)
  python scripts/clear_models.py --type hand_only    # Clear hand_only models only
  python scripts/clear_models.py --type opponent_field_only --force

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
    
    deleted = clear_models(models_dir, confirm=not args.force, training_type=args.type)
    
    if deleted > 0:
        # Optionally remove the directory if it's empty
        try:
            if models_dir.exists() and not any(models_dir.iterdir()):
                models_dir.rmdir()
                print(f"Removed empty directory: {models_dir}")
        except Exception as e:
            print(f"Note: Could not remove directory: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
