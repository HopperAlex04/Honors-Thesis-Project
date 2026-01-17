#!/usr/bin/env python3
"""
Master cleanup script that runs all cleanup utilities.

This script can clear logs, models, or both, providing a unified interface
for cleaning up generated files.

Usage:
    python scripts/cleanup.py                    # Interactive mode, clean all
    python scripts/cleanup.py --all              # Clean all (logs + models + training states)
    python scripts/cleanup.py --logs             # Clean logs only
    python scripts/cleanup.py --models           # Clean models only
    python scripts/cleanup.py --type hand_only   # Clean all for specific training type
    python scripts/cleanup.py --force            # No confirmation for all
    python scripts/cleanup.py --all --force      # Clean all without confirmation
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Import cleanup functions from other scripts
# Add scripts directory to path to allow imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from clear_logs import clear_logs
from clear_models import clear_models
from clear_training_state import clear_training_state

# Valid training types (shared across all scripts)
VALID_TRAINING_TYPES = [
    "hand_only",
    "opponent_field_only",
    "no_features",
    "both_features",  # Legacy name, kept for backward compatibility
    "all_features",
    "scores"
]


def cleanup_all(force: bool = False, training_type: Optional[str] = None) -> tuple[int, int, int]:
    """
    Run all cleanup scripts.
    
    Args:
        force: If True, skip confirmation prompts
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Tuple of (logs_deleted, models_deleted, training_states_deleted)
    """
    logs_deleted = 0
    models_deleted = 0
    training_states_deleted = 0
    
    print("=" * 60)
    print("CUTTLE PROJECT CLEANUP")
    if training_type:
        print(f"Training Type: {training_type}")
    else:
        print("Training Type: ALL")
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
        logs_deleted = clear_logs(log_dir, confirm=not force, training_type=training_type)
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
        models_deleted = clear_models(models_dir, confirm=not force, training_type=training_type)
    except Exception as e:
        print(f"Error cleaning models: {e}")
    
    print()
    
    # Clean training states
    print("─" * 60)
    print("CLEANING TRAINING STATES")
    print("─" * 60)
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        training_states_deleted = clear_training_state(models_dir, confirm=not force, training_type=training_type)
    except Exception as e:
        print(f"Error cleaning training states: {e}")
    
    print()
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Logs deleted:           {logs_deleted}")
    print(f"Models deleted:        {models_deleted}")
    print(f"Training states deleted: {training_states_deleted}")
    print(f"Total files deleted:   {logs_deleted + models_deleted + training_states_deleted}")
    print("=" * 60)
    
    return logs_deleted, models_deleted, training_states_deleted


def cleanup_logs_only(force: bool = False, training_type: Optional[str] = None) -> int:
    """
    Clean logs only.
    
    Args:
        force: If True, skip confirmation prompts
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of log files deleted
    """
    print("=" * 60)
    print("CLEANING LOGS")
    if training_type:
        print(f"Training Type: {training_type}")
    print("=" * 60)
    print()
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        log_dir = project_root / "action_logs"
        deleted = clear_logs(log_dir, confirm=not force, training_type=training_type)
        print()
        print(f"✓ Deleted {deleted} log file(s)")
        return deleted
    except Exception as e:
        print(f"Error cleaning logs: {e}")
        return 0


def cleanup_models_only(force: bool = False, training_type: Optional[str] = None) -> int:
    """
    Clean models only.
    
    Args:
        force: If True, skip confirmation prompts
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of model files deleted
    """
    print("=" * 60)
    print("CLEANING MODELS")
    if training_type:
        print(f"Training Type: {training_type}")
    print("=" * 60)
    print()
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        deleted = clear_models(models_dir, confirm=not force, training_type=training_type)
        print()
        print(f"✓ Deleted {deleted} model file(s)")
        return deleted
    except Exception as e:
        print(f"Error cleaning models: {e}")
        return 0


def cleanup_training_states_only(force: bool = False, training_type: Optional[str] = None) -> int:
    """
    Clean training states only.
    
    Args:
        force: If True, skip confirmation prompts
        training_type: Optional training type to filter by (e.g., "hand_only")
        
    Returns:
        Number of training state files deleted
    """
    print("=" * 60)
    print("CLEANING TRAINING STATES")
    if training_type:
        print(f"Training Type: {training_type}")
    print("=" * 60)
    print()
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        deleted = clear_training_state(models_dir, confirm=not force, training_type=training_type)
        print()
        print(f"✓ Deleted {deleted} training state file(s)")
        return deleted
    except Exception as e:
        print(f"Error cleaning training states: {e}")
        return 0


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Master cleanup script for Cuttle project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/cleanup.py                    # Interactive cleanup of all
  python scripts/cleanup.py --all              # Clean all (logs + models + training states)
  python scripts/cleanup.py --logs             # Clean logs only
  python scripts/cleanup.py --models           # Clean models only
  python scripts/cleanup.py --training-states  # Clean training states only
  python scripts/cleanup.py --type hand_only  # Clean all for specific training type
  python scripts/cleanup.py --type opponent_field_only --force
  python scripts/cleanup.py --logs --type hand_only  # Clean logs for specific type
  python scripts/cleanup.py --force            # Clean all without confirmation

Valid training types: {', '.join(VALID_TRAINING_TYPES)}
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
        "--training-states",
        action="store_true",
        help="Clean training states only"
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
    
    # Determine what to clean
    clean_logs = args.logs or args.all or (not args.logs and not args.models and not args.training_states and not args.all)
    clean_models = args.models or args.all or (not args.logs and not args.models and not args.training_states and not args.all)
    clean_training_states = args.training_states or args.all or (not args.logs and not args.models and not args.training_states and not args.all)
    
    # If specific flags are set, only clean those
    if args.logs and not args.all:
        clean_models = False
        clean_training_states = False
    if args.models and not args.all:
        clean_logs = False
        clean_training_states = False
    if args.training_states and not args.all:
        clean_logs = False
        clean_models = False
    
    # Run cleanup
    if clean_logs and clean_models and clean_training_states:
        cleanup_all(force=args.force, training_type=args.type)
    elif clean_logs and not clean_models and not clean_training_states:
        cleanup_logs_only(force=args.force, training_type=args.type)
    elif clean_models and not clean_logs and not clean_training_states:
        cleanup_models_only(force=args.force, training_type=args.type)
    elif clean_training_states and not clean_logs and not clean_models:
        cleanup_training_states_only(force=args.force, training_type=args.type)
    else:
        # Default: clean all
        cleanup_all(force=args.force, training_type=args.type)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

