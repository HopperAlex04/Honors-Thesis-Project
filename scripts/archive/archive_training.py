#!/usr/bin/env python3
"""
Archive all training-related files including models, logs, graphs, and checkpoints.

This script creates a timestamped archive directory containing:
- All model checkpoints (.pt files)
- Training state files (training_state_*.json)
- Action logs (action_logs/)
- Metrics graphs (metrics_graphs/)
- Improvement analysis graphs (improvement_analysis/)
- Training configuration (hyperparams_config.json)
- Analysis documentation (optional)

Usage:
    python scripts/archive_training.py                    # Interactive mode
    python scripts/archive_training.py --output archives/  # Specify output directory
    python scripts/archive_training.py --compress          # Create compressed archive
    python scripts/archive_training.py --force             # No confirmation prompts
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_directory_size(directory: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total_size = 0
    if directory.exists() and directory.is_dir():
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size


def get_file_count(directory: Path) -> int:
    """Count number of files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob('*') if _.is_file())


def create_archive_manifest(project_root: Path, archive_root: Path) -> dict:
    """Create a manifest of all archived files and metadata."""
    manifest = {
        "archive_date": datetime.now().isoformat(),
        "archive_version": "1.0",
        "project_root": str(project_root.absolute()),
        "contents": {}
    }
    
    # Models directory
    models_dir = project_root / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        state_files = list(models_dir.glob("training_state*.json"))
        manifest["contents"]["models"] = {
            "checkpoints": len(model_files),
            "state_files": len(state_files),
            "files": [f.name for f in sorted(model_files + state_files)]
        }
    
    # Action logs
    action_logs_dir = project_root / "action_logs"
    if action_logs_dir.exists():
        file_count = get_file_count(action_logs_dir)
        size = get_directory_size(action_logs_dir)
        manifest["contents"]["action_logs"] = {
            "file_count": file_count,
            "size_bytes": size,
            "subdirectories": [d.name for d in action_logs_dir.iterdir() if d.is_dir()]
        }
    
    # Metrics graphs
    metrics_graphs_dir = project_root / "metrics_graphs"
    if metrics_graphs_dir.exists():
        file_count = get_file_count(metrics_graphs_dir)
        size = get_directory_size(metrics_graphs_dir)
        manifest["contents"]["metrics_graphs"] = {
            "file_count": file_count,
            "size_bytes": size,
            "subdirectories": [d.name for d in metrics_graphs_dir.iterdir() if d.is_dir()]
        }
    
    # Improvement analysis
    improvement_analysis_dir = project_root / "improvement_analysis"
    if improvement_analysis_dir.exists():
        files = list(improvement_analysis_dir.glob("*.png"))
        size = sum(f.stat().st_size for f in files)
        manifest["contents"]["improvement_analysis"] = {
            "file_count": len(files),
            "size_bytes": size,
            "files": [f.name for f in sorted(files)]
        }
    
    # Config file
    config_file = project_root / "hyperparams_config.json"
    if config_file.exists():
        manifest["contents"]["config"] = {
            "file": config_file.name,
            "size_bytes": config_file.stat().st_size
        }
    
    # Training scripts (optional - just for reference)
    training_scripts = [
        "train_no_features.py",
        "train_hand_feature_only.py",
        "train_opponent_field_only.py",
        "train_both_features.py",
        "train_scores.py"
    ]
    found_scripts = [s for s in training_scripts if (project_root / s).exists()]
    if found_scripts:
        manifest["contents"]["training_scripts"] = {
            "files": found_scripts
        }
    
    return manifest


def archive_training_data(
    project_root: Path,
    output_dir: Path,
    compress: bool = False,
    include_docs: bool = False
) -> Path:
    """
    Archive all training-related data to a timestamped directory.
    
    Args:
        project_root: Root directory of the project
        output_dir: Directory where archive should be created
        compress: If True, compress the archive into a tar.gz file
        include_docs: If True, include analysis documentation files
        
    Returns:
        Path to the archive directory or compressed file
    """
    # Create timestamped archive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"training_archive_{timestamp}"
    archive_path = output_dir / archive_name
    
    print(f"Creating archive: {archive_path}")
    print()
    
    # Create archive directory
    archive_path.mkdir(parents=True, exist_ok=True)
    
    total_size = 0
    copied_items = []
    
    # 1. Archive models directory (checkpoints and state files)
    models_dir = project_root / "models"
    if models_dir.exists() and models_dir.is_dir():
        print("Archiving models...")
        archive_models = archive_path / "models"
        archive_models.mkdir(exist_ok=True)
        
        # Copy all .pt files
        model_files = list(models_dir.glob("*.pt"))
        for model_file in model_files:
            shutil.copy2(model_file, archive_models / model_file.name)
            size = model_file.stat().st_size
            total_size += size
            copied_items.append(f"models/{model_file.name}")
        
        # Copy training state files
        state_files = list(models_dir.glob("training_state*.json"))
        for state_file in state_files:
            shutil.copy2(state_file, archive_models / state_file.name)
            size = state_file.stat().st_size
            total_size += size
            copied_items.append(f"models/{state_file.name}")
        
        print(f"  ✓ Copied {len(model_files)} checkpoints and {len(state_files)} state files")
    
    # 2. Archive action logs
    action_logs_dir = project_root / "action_logs"
    if action_logs_dir.exists() and action_logs_dir.is_dir():
        print("Archiving action logs...")
        archive_action_logs = archive_path / "action_logs"
        shutil.copytree(action_logs_dir, archive_action_logs, dirs_exist_ok=True)
        size = get_directory_size(action_logs_dir)
        total_size += size
        file_count = get_file_count(action_logs_dir)
        copied_items.append(f"action_logs/ ({file_count} files)")
        print(f"  ✓ Copied {file_count} files")
    
    # 3. Archive metrics graphs
    metrics_graphs_dir = project_root / "metrics_graphs"
    if metrics_graphs_dir.exists() and metrics_graphs_dir.is_dir():
        print("Archiving metrics graphs...")
        archive_metrics = archive_path / "metrics_graphs"
        shutil.copytree(metrics_graphs_dir, archive_metrics, dirs_exist_ok=True)
        size = get_directory_size(metrics_graphs_dir)
        total_size += size
        file_count = get_file_count(metrics_graphs_dir)
        copied_items.append(f"metrics_graphs/ ({file_count} files)")
        print(f"  ✓ Copied {file_count} files")
    
    # 4. Archive improvement analysis
    improvement_analysis_dir = project_root / "improvement_analysis"
    if improvement_analysis_dir.exists() and improvement_analysis_dir.is_dir():
        print("Archiving improvement analysis...")
        archive_improvement = archive_path / "improvement_analysis"
        archive_improvement.mkdir(exist_ok=True)
        
        png_files = list(improvement_analysis_dir.glob("*.png"))
        for png_file in png_files:
            shutil.copy2(png_file, archive_improvement / png_file.name)
            total_size += png_file.stat().st_size
            copied_items.append(f"improvement_analysis/{png_file.name}")
        
        print(f"  ✓ Copied {len(png_files)} analysis graphs")
    
    # 5. Archive config file
    config_file = project_root / "hyperparams_config.json"
    if config_file.exists():
        print("Archiving configuration...")
        shutil.copy2(config_file, archive_path / "hyperparams_config.json")
        total_size += config_file.stat().st_size
        copied_items.append("hyperparams_config.json")
        print("  ✓ Copied config file")
    
    # 6. Optionally archive analysis documentation
    if include_docs:
        print("Archiving analysis documentation...")
        doc_files = [
            "HYPERPARAMETER_EXPERIMENTATION.md",
            "NETWORK_CAPACITY_ANALYSIS.md",
            "REGRESSION_SOURCES_ANALYSIS.md",
            "RESEARCH_METHODOLOGY_LOSS_VS_PERFORMANCE.md",
            "ROOT_CAUSE_ANALYSIS.md",
            "ROUNDS_VS_EPISODES_ANALYSIS.md"
        ]
        docs_archived = []
        for doc_file in doc_files:
            doc_path = project_root / doc_file
            if doc_path.exists():
                shutil.copy2(doc_path, archive_path / doc_file)
                total_size += doc_path.stat().st_size
                docs_archived.append(doc_file)
                copied_items.append(doc_file)
        if docs_archived:
            print(f"  ✓ Copied {len(docs_archived)} documentation files")
        else:
            print("  - No documentation files found")
    
    # 7. Create manifest
    print("Creating archive manifest...")
    manifest = create_archive_manifest(project_root, archive_path)
    manifest_path = archive_path / "ARCHIVE_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    copied_items.append("ARCHIVE_MANIFEST.json")
    print("  ✓ Created manifest")
    
    # 8. Create README
    readme_path = archive_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Training Archive - {timestamp}

This archive contains all training-related files from the Cuttle Card Game DQN project.

## Archive Information

- **Archive Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Size:** {format_size(total_size)}
- **Source:** {project_root.absolute()}

## Contents

""")
        for item in copied_items:
            f.write(f"- {item}\n")
        
        f.write(f"""
## Directory Structure

- `models/` - Model checkpoints (.pt files) and training state files
- `action_logs/` - Training action logs organized by feature configuration
- `metrics_graphs/` - Training metrics and evaluation graphs
- `improvement_analysis/` - Analysis graphs comparing model improvements
- `hyperparams_config.json` - Training hyperparameters configuration
- `ARCHIVE_MANIFEST.json` - Detailed manifest of archived files

## Restoring This Archive

To restore this archive:

1. Extract to your project root (if compressed)
2. Copy files back to their original locations:
   ```bash
   cp -r models/* /path/to/project/models/
   cp -r action_logs/* /path/to/project/action_logs/
   cp -r metrics_graphs/* /path/to/project/metrics_graphs/
   cp -r improvement_analysis/* /path/to/project/improvement_analysis/
   cp hyperparams_config.json /path/to/project/
   ```

## Version Information

This archive was created using version 1.0.0 of the archiving script.
""")
    
    print()
    print("="*60)
    print(f"Archive created successfully!")
    print(f"Location: {archive_path.absolute()}")
    print(f"Total size: {format_size(total_size)}")
    print(f"Items archived: {len(copied_items)}")
    print("="*60)
    
    # Compress if requested
    if compress:
        print()
        print("Compressing archive...")
        compressed_path = output_dir / f"{archive_name}.tar.gz"
        shutil.make_archive(
            str(output_dir / archive_name),
            'gztar',
            root_dir=output_dir,
            base_dir=archive_name
        )
        compressed_size = compressed_path.stat().st_size
        print(f"  ✓ Compressed archive: {compressed_path}")
        print(f"  Compressed size: {format_size(compressed_size)}")
        print(f"  Compression ratio: {(1 - compressed_size / total_size) * 100:.1f}%")
        print()
        print(f"Compressed archive: {compressed_path.absolute()}")
        return compressed_path
    
    return archive_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Archive all training-related files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/archive_training.py                    # Interactive mode
  python scripts/archive_training.py --output archives/  # Custom output directory
  python scripts/archive_training.py --compress          # Create compressed archive
  python scripts/archive_training.py --include-docs      # Include documentation
  python scripts/archive_training.py --force             # No confirmation prompts
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="archives",
        help="Output directory for archives (default: archives)"
    )
    parser.add_argument(
        "--compress", "-c",
        action="store_true",
        help="Compress archive into tar.gz file"
    )
    parser.add_argument(
        "--include-docs", "-d",
        action="store_true",
        help="Include analysis documentation files"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Create output directory
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Training Data Archiver")
    print("="*60)
    print(f"Project root: {project_root.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Compress: {'Yes' if args.compress else 'No'}")
    print(f"Include docs: {'Yes' if args.include_docs else 'No'}")
    print("="*60)
    print()
    
    # Confirmation
    if not args.force:
        response = input("Create archive? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    # Create archive
    try:
        archive_path = archive_training_data(
            project_root=project_root,
            output_dir=output_dir,
            compress=args.compress,
            include_docs=args.include_docs
        )
        print()
        print("✓ Archive complete!")
        return 0
    except Exception as e:
        print(f"\n✗ Error creating archive: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

