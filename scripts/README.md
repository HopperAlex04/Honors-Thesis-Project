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

