# Scripts

Utility scripts for the Cuttle project.

## Experiment Management (Full Experiment)

For running the full input representation comparison experiment (3 network types × 7 runs = 21 total runs), see **[EXPERIMENT_MANAGEMENT.md](EXPERIMENT_MANAGEMENT.md)** for detailed documentation.

### Quick Start

```bash
# Initialize experiment
python scripts/experiment_manager.py init --name "input_rep_v1"

# Run all training sessions
python scripts/run_full_experiment.py --parallel 3

# Analyze results
python scripts/aggregate_experiment_results.py --graphs

# Generate detailed graphs
python scripts/generate_metrics_graphs.py --all
```

### Experiment Scripts

| Script | Purpose |
|--------|---------|
| `experiment_manager.py` | Initialize, track, and manage experiments |
| `run_full_experiment.py` | Execute training runs (sequential or parallel) |
| `aggregate_experiment_results.py` | Analyze and visualize results |
| `generate_metrics_graphs.py` | Generate round-level and episode-level graphs |
| `migrate_experiment_logs.py` | Migrate legacy experiments to new log structure |

### Directory Structure

Each experiment run has separate directories for different log types:
```
runs/<run_id>/
├── action_logs/    # Per-turn action data (for strategy analysis)
├── metrics_logs/   # Per-episode metrics (for graphing)
├── models/         # Model checkpoints
├── hyperparams_config.json
└── run_metadata.json
```

---

## cleanup.py

Master cleanup script that runs all cleanup utilities. This is the recommended way to clean up generated files.

### Usage

```bash
# Interactive mode - clean all (logs + models)
python scripts/cleanup.py

# Clean all without confirmation
python scripts/cleanup.py --all --force
python scripts/cleanup.py --force

# Clean logs only
python scripts/cleanup.py --logs
python scripts/cleanup.py --logs --force

# Clean models only
python scripts/cleanup.py --models
python scripts/cleanup.py --models --force
```

### What it does

- Provides a unified interface for all cleanup operations
- Can clean logs, models, or both
- Shows a summary of all deletions
- Respects the `--force` flag to skip confirmations

### Examples

```bash
# Clean everything interactively
python scripts/cleanup.py

# Clean everything without prompts
python scripts/cleanup.py --all --force

# Clean only logs
python scripts/cleanup.py --logs

# Clean only models (with confirmation)
python scripts/cleanup.py --models
```

## clear_logs.py

Clears all training logs from the `action_logs/` directory.

### Usage

```bash
# Interactive mode (asks for confirmation)
python scripts/clear_logs.py

# Force mode (no confirmation)
python scripts/clear_logs.py --force
python scripts/clear_logs.py -f
```

### What it does

- Finds all `actions_*.jsonl` and `metrics_*.jsonl` files in `action_logs/`
- Shows file count and sizes
- Asks for confirmation (unless `--force` is used)
- Deletes all log files
- Removes the directory if it becomes empty

## clear_models.py

Clears all model checkpoints from the `models/` directory.

### Usage

```bash
# Interactive mode (asks for confirmation)
python scripts/clear_models.py

# Force mode (no confirmation)
python scripts/clear_models.py --force
python scripts/clear_models.py -f
```

### What it does

- Finds all model files (`.pt`, `.pth`) and checkpoint files in `models/`
- Shows file count, individual sizes, and total size
- Asks for confirmation (unless `--force` is used)
- Deletes all model files
- Removes the directory if it becomes empty

### Warning

**WARNING: This will delete all saved model checkpoints!** Make sure you have backups if needed.

## archive_training.py

Archives all training-related files including models, logs, graphs, and checkpoints into a timestamped directory.

### Usage

```bash
# Interactive mode - create archive in default 'archives/' directory
python scripts/archive_training.py

# Specify custom output directory
python scripts/archive_training.py --output archives/

# Create compressed archive (tar.gz)
python scripts/archive_training.py --compress

# Include analysis documentation files
python scripts/archive_training.py --include-docs

# No confirmation prompts
python scripts/archive_training.py --force
```

### What it archives

- **models/** - All model checkpoints (.pt files) and training state files (training_state_*.json)
- **action_logs/** - All training action logs organized by feature configuration
- **metrics_graphs/** - All training metrics and evaluation graphs
- **improvement_analysis/** - Analysis graphs comparing model improvements
- **hyperparams_config.json** - Training hyperparameters configuration
- **ARCHIVE_MANIFEST.json** - Detailed manifest of archived files (auto-generated)
- **README.md** - Archive information and restoration instructions (auto-generated)
- **Documentation** (optional with `--include-docs`) - Analysis markdown files

### Examples

```bash
# Create basic archive
python scripts/archive_training.py

# Create compressed archive with documentation
python scripts/archive_training.py --compress --include-docs

# Archive to custom location without prompts
python scripts/archive_training.py --output ~/backups --force
```

### Archive Structure

Each archive is created with a timestamped name: `training_archive_YYYYMMDD_HHMMSS/`

The archive contains:
- All training files organized in their original directory structure
- A manifest file (JSON) with detailed information about archived contents
- A README with restoration instructions

### Restoration

To restore an archive, copy files back to their original locations:
```bash
cp -r archives/training_archive_YYYYMMDD_HHMMSS/models/* models/
cp -r archives/training_archive_YYYYMMDD_HHMMSS/action_logs/* action_logs/
cp -r archives/training_archive_YYYYMMDD_HHMMSS/metrics_graphs/* metrics_graphs/
cp -r archives/training_archive_YYYYMMDD_HHMMSS/improvement_analysis/* improvement_analysis/
cp archives/training_archive_YYYYMMDD_HHMMSS/hyperparams_config.json .
```

