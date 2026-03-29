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
    │   │   ├── model_round_0.pt … model_round_N.pt  (if round checkpointing)
    │   │   ├── model_best.pt, best_round.json      (if round checkpointing)
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

## Experiment creation scripts

**Active (project root):** `create_double_dqn_peak_conditions.py`, `create_ppo_experiment.py`, and `experiment_manager.py`.

**Archived** generators (scaling, scale-11 vs large hidden, self-play, gapmaximizer-focused Double DQN, etc.) live in `archive/experiment_creation_scripts/`. See that folder’s `README.txt`. Run them with an explicit path, for example:

```bash
python archive/experiment_creation_scripts/create_scale11_vs_large_hidden_experiment.py
python experiment_manager.py run
```

## Round checkpointing

Training now saves **per-round checkpoints** and tracks the **best** model by validation (vs GapMaximizer when available):

- **model_round_0.pt … model_round_N.pt** — checkpoint after each round (state_dict, optimizer, config, round, win_rate_history).
- **model_best.pt** — copy of the round with the best validation win rate (vs GapMaximizer if present, else first opponent).
- **best_round.json** — `{ "best_round": N, "best_win_rate": float, "best_metric_opponent": "GapMaximizer" }`.

Use **model_best.pt** or **model_round_N.pt** for evaluation instead of only **model_final.pt** when you want the best validation checkpoint. Per-run **rounds** can be overridden in experiment run config (e.g. 5 rounds for short comparison experiments).

## Notes

- All runs execute **sequentially** (no parallel execution)
- Outputs are organized per run for easy analysis
- Progress is tracked in `runs_status.json`
- Final metrics are extracted from the latest metrics log files
