#!/usr/bin/env python3
"""
Script to clear model checkpoints from the models directory.

This script removes all model files (.pt, .pth) and checkpoint files
from the models directory.

Usage:
    python scripts/clear_models.py          # Interactive mode
    python scripts/clear_models.py --force  # No confirmation
"""

import sys
from pathlib import Path
from typing import List


def get_model_files(models_dir: Path) -> List[Path]:
    """
    Get all model files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        List of model file paths
    """
    if not models_dir.exists():
        return []
    
    model_files = []
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


def clear_models(models_dir: Path, confirm: bool = True) -> int:
    """
    Clear all model files from the models directory.
    
    Args:
        models_dir: Path to the models directory
        confirm: If True, ask for confirmation before deleting
        
    Returns:
        Number of files deleted
    """
    model_files = get_model_files(models_dir)
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return 0
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_files)
    
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
    # Get the project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Models directory is in project root
    models_dir = project_root / "models"
    
    # Check for command line arguments
    force = "--force" in sys.argv or "-f" in sys.argv
    
    print(f"Models directory: {models_dir.absolute()}")
    print()
    
    deleted = clear_models(models_dir, confirm=not force)
    
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
