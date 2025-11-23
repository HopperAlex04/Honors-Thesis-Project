# Scripts

Utility scripts for the Cuttle project.

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

⚠️ **This will delete all saved model checkpoints!** Make sure you have backups if needed.

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

- Finds all model files (`*.pt`, `*.pth`, `*.pkl`, `*.ckpt`) in `models/`
- Shows file count, names, and sizes (with total size)
- Asks for confirmation (unless `--force` is used)
- Deletes all model files
- Removes the directory if it becomes empty

