# Scripts

Utility scripts for the Cuttle project.

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

