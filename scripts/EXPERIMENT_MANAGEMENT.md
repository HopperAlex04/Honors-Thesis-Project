# Experiment Management System

This document describes the experiment management system for running the full input representation comparison experiment (3 network types × 7 runs = 21 total runs).

## Overview

The experiment management system consists of three main scripts:

1. **`experiment_manager.py`** - Core management: initialize, track status, manage runs
2. **`run_full_experiment.py`** - Execute training runs (sequential or parallel)
3. **`aggregate_experiment_results.py`** - Analyze and visualize results

## Quick Start

### 1. Initialize an Experiment

```bash
# Create a new experiment
python scripts/experiment_manager.py init --name "input_rep_v1" --description "Comparing Boolean, Embedding, and Multi-Encoder networks"

# This creates:
# - experiments/experiment_YYYYMMDD_HHMMSS_input_rep_v1/
#   ├── runs/                    # Individual run directories
#   ├── analysis/                # Results and graphs
#   ├── base_hyperparams_config.json
#   ├── experiment_metadata.json
#   └── runs_status.json
```

### 2. Check Status

```bash
# View current experiment status
python scripts/experiment_manager.py status

# List all experiments
python scripts/experiment_manager.py list

# Select a different experiment
python scripts/experiment_manager.py select input_rep_v1
```

### 3. Run the Experiment

#### Option A: Run All at Once (Sequential)
```bash
python scripts/run_full_experiment.py
```

#### Option B: Run with Parallelization
```bash
# Run up to 3 training sessions in parallel
python scripts/run_full_experiment.py --parallel 3
```

#### Option C: Run Specific Types or Runs
```bash
# Run only boolean network experiments
python scripts/run_full_experiment.py --type boolean

# Run specific run numbers (e.g., runs 1, 2, 3)
python scripts/run_full_experiment.py --runs 1,2,3

# Run a single specific run
python scripts/experiment_manager.py run --type boolean --run 1
```

### 4. Analyze Results

```bash
# View aggregated statistics
python scripts/aggregate_experiment_results.py

# Generate comparison graphs
python scripts/aggregate_experiment_results.py --graphs

# Export results
python scripts/aggregate_experiment_results.py --export csv
python scripts/aggregate_experiment_results.py --export latex
```

## Directory Structure

After running experiments, your structure will look like:

```
experiments/
└── experiment_20260119_input_rep_v1/
    ├── runs/
    │   ├── boolean_run_01/
    │   │   ├── models/
    │   │   │   ├── model_initial.pt
    │   │   │   └── model_final.pt
    │   │   ├── action_logs/
    │   │   │   └── ...
    │   │   ├── hyperparams_config.json
    │   │   └── run_metadata.json
    │   ├── boolean_run_02/
    │   ├── ...
    │   ├── embedding_run_01/
    │   ├── ...
    │   └── multi_encoder_run_07/
    ├── analysis/
    │   ├── graphs/
    │   │   ├── win_rate_comparison.png
    │   │   ├── training_curves.png
    │   │   └── win_rate_distribution.png
    │   ├── full_analysis.json
    │   └── statistics.json
    ├── base_hyperparams_config.json
    ├── experiment_metadata.json
    └── runs_status.json
```

## Configuration

### Experiment Settings

The experiment uses these defaults (configurable in `experiment_manager.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| Network Types | boolean, embedding, multi_encoder | Network architectures to compare |
| Runs per Type | 7 | Number of independent runs for statistical significance |
| Total Runs | 21 | Total training runs (3 × 7) |

### Per-Run Configuration

Each run gets its own configuration with:
- **Unique seed**: Deterministically generated for reproducibility
- **Network type**: Set automatically based on run ID
- **Quick test mode**: Disabled for real experiments

### Base Hyperparameters

Edit `hyperparams_config.json` before initializing an experiment to set base hyperparameters:

```json
{
  "training": {
    "rounds": 20,
    "eps_per_round": 250,
    "quick_test_mode": false,
    "validation_episodes_ratio": 0.5,
    "validation_opponent": "both"
  }
}
```

## Commands Reference

### experiment_manager.py

| Command | Description |
|---------|-------------|
| `init --name NAME` | Initialize a new experiment |
| `status` | Show current experiment status |
| `list` | List all experiments |
| `select NAME` | Select an experiment as current |
| `run --type TYPE --run N` | Execute a single run |
| `analyze` | Analyze completed runs |
| `export --format FMT` | Export data (csv, json) |

### run_full_experiment.py

| Option | Description |
|--------|-------------|
| `--parallel N` | Run N training sessions in parallel |
| `--type TYPE` | Only run specific network type |
| `--runs 1,2,3` | Only run specific run numbers |
| `--dry-run` | Show what would run without executing |
| `--force` | Re-run completed runs |
| `--init NAME` | Initialize experiment before running |

### aggregate_experiment_results.py

| Option | Description |
|--------|-------------|
| `--graphs` | Generate comparison visualizations |
| `--export FORMAT` | Export results (json, csv, latex) |
| `--output-dir PATH` | Custom output directory |

## Statistical Analysis

The aggregation script calculates:

1. **Descriptive Statistics** (per network type):
   - Mean win rate
   - Standard deviation
   - 95% confidence interval
   - Min/max/median

2. **Statistical Tests**:
   - Pairwise t-tests (with normality check)
   - Welch's t-test (if unequal variance)
   - Mann-Whitney U (if non-normal)
   - Bonferroni correction for multiple comparisons
   - ANOVA across all groups
   - Effect sizes (Cohen's d, η²)

## Generated Graphs

The system generates thesis-quality visualizations:

1. **win_rate_comparison.png**: Bar chart with 95% CI error bars and individual data points
2. **training_curves.png**: Learning curves showing mean ± std across runs
3. **win_rate_distribution.png**: Box plots with individual points

## Time Estimates

Based on your training time estimates:

| Scenario | Per Run | All 21 Runs (Sequential) | Parallel (3) |
|----------|---------|-------------------------|--------------|
| Optimistic | ~45 min | ~16 hours | ~5 hours |
| Average | ~1 hour | ~21 hours | ~7 hours |
| Pessimistic | ~1.5 hours | ~32 hours | ~11 hours |

## Storage Estimates

| Component | Per Run | All 21 Runs |
|-----------|---------|-------------|
| Models | ~1-2 GB | ~21-42 GB |
| Logs | ~0.5-1 GB | ~10-21 GB |
| **Total** | ~1.5-3 GB | ~31-63 GB |

## Tips

1. **Start with a test run**: Before running all 21, test with one run:
   ```bash
   python scripts/experiment_manager.py run --type boolean --run 1
   ```

2. **Monitor progress**: Check status periodically:
   ```bash
   python scripts/experiment_manager.py status
   ```

3. **Resume after interruption**: The system tracks run status. Completed runs are skipped:
   ```bash
   python scripts/run_full_experiment.py  # Will skip completed runs, restart interrupted ones
   ```
   Note: If a run is interrupted mid-training, it will restart from the beginning (not resume mid-run).

4. **Parallel considerations**:
   - Parallelization uses more memory and CPU
   - 2-3 parallel is usually safe on modern machines
   - More than 3 may cause thermal throttling

5. **Archive when done**: After analysis, archive the experiment:
   ```bash
   tar -czvf experiment_backup.tar.gz experiments/experiment_YYYYMMDD_*/
   ```

## Troubleshooting

### Run Failed

Check the error message:
```bash
python scripts/experiment_manager.py status
```

Re-run the specific failed run:
```bash
python scripts/experiment_manager.py run --type TYPE --run N --force
```

### Missing scipy for Statistics

Install scipy for full statistical analysis:
```bash
pip install scipy
```

### Out of Disk Space

Each run only saves initial and final models, so disk usage should be manageable.
To clean up action logs from completed runs:
```bash
find experiments/*/runs/*/action_logs -name "*.jsonl" -delete
```
