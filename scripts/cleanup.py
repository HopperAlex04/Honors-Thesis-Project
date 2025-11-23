#!/usr/bin/env python3
"""
Master cleanup script that runs all cleanup utilities.

This script can clear logs, models, or both, providing a unified interface
for cleaning up generated files.

Usage:
    python scripts/cleanup.py                    # Interactive mode, clean all
    python scripts/cleanup.py --all              # Clean all (logs + models)
    python scripts/cleanup.py --logs             # Clean logs only
    python scripts/cleanup.py --models           # Clean models only
    python scripts/cleanup.py --force            # No confirmation for all
    python scripts/cleanup.py --all --force      # Clean all without confirmation
"""

import sys
import argparse
from pathlib import Path

# Import cleanup functions from other scripts
# Add scripts directory to path to allow imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from clear_logs import clear_logs
from clear_models import clear_models


def cleanup_all(force: bool = False) -> tuple[int, int]:
    """
    Run all cleanup scripts.
    
    Args:
        force: If True, skip confirmation prompts
        
    Returns:
        Tuple of (logs_deleted, models_deleted)
    """
    logs_deleted = 0
    models_deleted = 0
    
    print("=" * 60)
    print("CUTTLE PROJECT CLEANUP")
    print("=" * 60)
    print()
    
    # Clean logs
    print("─" * 60)
    print("CLEANING LOGS")
    print("─" * 60)
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        log_dir = project_root / "action_logs"
        logs_deleted = clear_logs(log_dir, confirm=not force)
    except Exception as e:
        print(f"Error cleaning logs: {e}")
    
    print()
    
    # Clean models
    print("─" * 60)
    print("CLEANING MODELS")
    print("─" * 60)
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        models_deleted = clear_models(models_dir, confirm=not force)
    except Exception as e:
        print(f"Error cleaning models: {e}")
    
    print()
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Logs deleted:  {logs_deleted}")
    print(f"Models deleted: {models_deleted}")
    print(f"Total files deleted: {logs_deleted + models_deleted}")
    print("=" * 60)
    
    return logs_deleted, models_deleted


def cleanup_logs_only(force: bool = False) -> int:
    """
    Clean logs only.
    
    Args:
        force: If True, skip confirmation prompts
        
    Returns:
        Number of log files deleted
    """
    print("=" * 60)
    print("CLEANING LOGS")
    print("=" * 60)
    print()
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        log_dir = project_root / "action_logs"
        deleted = clear_logs(log_dir, confirm=not force)
        print()
        print(f"✓ Deleted {deleted} log file(s)")
        return deleted
    except Exception as e:
        print(f"Error cleaning logs: {e}")
        return 0


def cleanup_models_only(force: bool = False) -> int:
    """
    Clean models only.
    
    Args:
        force: If True, skip confirmation prompts
        
    Returns:
        Number of model files deleted
    """
    print("=" * 60)
    print("CLEANING MODELS")
    print("=" * 60)
    print()
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        deleted = clear_models(models_dir, confirm=not force)
        print()
        print(f"✓ Deleted {deleted} model file(s)")
        return deleted
    except Exception as e:
        print(f"Error cleaning models: {e}")
        return 0


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Master cleanup script for Cuttle project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup.py              # Interactive cleanup of all
  python scripts/cleanup.py --all        # Clean all (logs + models)
  python scripts/cleanup.py --logs       # Clean logs only
  python scripts/cleanup.py --models     # Clean models only
  python scripts/cleanup.py --force      # Clean all without confirmation
  python scripts/cleanup.py --logs --force  # Clean logs without confirmation
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean all (logs and models). This is the default if no specific target is specified."
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clean logs only"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Clean models only"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Determine what to clean
    clean_logs = args.logs or args.all or (not args.logs and not args.models and not args.all)
    clean_models = args.models or args.all or (not args.logs and not args.models and not args.all)
    
    # If specific flags are set, only clean those
    if args.logs and not args.all:
        clean_models = False
    if args.models and not args.all:
        clean_logs = False
    
    # Run cleanup
    if clean_logs and clean_models:
        cleanup_all(force=args.force)
    elif clean_logs:
        cleanup_logs_only(force=args.force)
    elif clean_models:
        cleanup_models_only(force=args.force)
    else:
        # Default: clean all
        cleanup_all(force=args.force)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

