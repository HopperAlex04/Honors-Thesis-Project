# Experiment Manager

Simple experiment manager for running architecture comparison experiments.

## Overview

Runs 6 configurations sequentially:
- **linear** with boolean input
- **linear** with embedding input
- **large_hidden** (512 neurons) with boolean input
- **large_hidden** (512 neurons) with embedding input
- **game_based** ([104, 26, 30] layers) with boolean input
- **game_based** ([104, 26, 30] layers) with embedding input

Each configuration runs 5 times with different random seeds for statistical significance.

**Total: 30 runs** (6 configurations × 5 runs each)

## Usage

### 1. Initialize an Experiment

```bash
python experiment_manager.py init --name "architecture_comparison"
```

This creates:
- Experiment directory in `experiments/`
- Configuration files for all 30 runs
- Run tracking metadata

### 2. Run the Experiment

```bash
python experiment_manager.py run
```

This will:
- Run all pending runs sequentially
- Organize outputs (models, action_logs, metrics_logs) per run
- Track progress and status
- Extract final metrics

### 3. Check Status

```bash
python experiment_manager.py status
```

Shows:
- Overall experiment status
- Runs by status (pending, running, completed, failed)
- Progress by configuration

## Output Organization

Each run creates its own directory structure:

```
experiments/experiment_YYYYMMDD_HHMMSS_name/
├── experiment_metadata.json
├── runs_status.json
├── base_hyperparams_config.json
└── runs/
    ├── linear_boolean_run_01/
    │   ├── hyperparams_config.json
    │   ├── run_metadata.json
    │   ├── models/
    │   │   ├── model_initial.pt
    │   │   └── model_final.pt
    │   ├── action_logs/
    │   │   └── actions_*.jsonl
    │   └── metrics_logs/
    │       └── metrics_round_*_*.jsonl
    ├── linear_embedding_run_01/
    │   └── ...
    └── ...
```

## Configuration

The experiment uses `hyperparams_config.json` as the base configuration. Each run:
- Sets `network_type` (linear, large_hidden, game_based)
- Sets `use_embeddings` (true for embedding, false for boolean)
- Sets `random_seed` (unique per run)

All other hyperparameters remain constant across runs for fair comparison.

## Time Estimates

Based on previous runs:
- **Boolean runs**: ~1.67 hours each
- **Embedding runs**: ~1.25 hours each
- **Total sequential time**: ~43.8 hours (~1.8 days)

Runs execute one at a time, so you can monitor progress and handle any issues.

## Re-running Failed Runs

If a run fails, you can re-run it:

```bash
python experiment_manager.py run --force
```

This will re-run all runs (including completed ones). To re-run only failed runs, edit `runs_status.json` to set their status back to "pending".

## Scaling Experiments

### Original Scaling (create_scaling_experiment.py)

Creates experiments scaling game_based `[52k, 13k, 15k]` from k=1 upward:

```bash
python create_scaling_experiment.py --name "game_based_scaling" [--baseline-experiment EXPERIMENT]
```

### Reversed Scaling (create_reversed_scaling_experiment.py)

Creates experiments with **reversed** layer order `[15k, 13k, 52k]` (narrow→wide). Scales **down** from the parameter count that matches large_hidden (k≈11):

```bash
python create_reversed_scaling_experiment.py --name "game_based_reversed_scaling" [--baseline-experiment EXPERIMENT]
```

- Layer order: `[15k, 13k, 52k]` (reversed from original)
- Scales: k=11 down to k=1 (descending from large_hidden match)
- Both boolean and embedding inputs
- 1 run per config (screening)

## Notes

- All runs execute **sequentially** (no parallel execution)
- Outputs are organized per run for easy analysis
- Progress is tracked in `runs_status.json`
- Final metrics are extracted from the latest metrics log files
