# Metrics Graph Generation Script

## Quick Start

Generate graphs for the current experiment:
```bash
python scripts/generate_metrics_graphs.py
```

Generate all graph types (round-level and episode-level):
```bash
python scripts/generate_metrics_graphs.py --all
```

Generate graphs for a specific experiment:
```bash
python scripts/generate_metrics_graphs.py --experiment sparse_rewards
```

Compare multiple experiments:
```bash
python scripts/generate_metrics_graphs.py --compare sparse_rewards intermediate_rewards
```

## Features

### Primary Features
- Works directly with the experiment manager directory structure
- Supports both round-level and episode-level visualizations
- Automatically finds metrics in `metrics_logs/` (or `action_logs/` for legacy)
- Aggregates across multiple runs with mean and standard deviation
- Compare multiple experiments side-by-side

### Graph Types

#### Round-Level Graphs (`--rounds`, default)
- Win rate by training round vs validation opponents
- Shows mean line with confidence band
- Individual run traces shown with low opacity

#### Episode-Level Graphs (`--episodic`)
- Training loss over episodes (smoothed)
- Epsilon decay (exploration schedule)
- Raw data shown when dataset is small

#### Comparison Graphs (`--compare`)
- Side-by-side comparison of multiple experiments
- Same scale for fair comparison

## Command-Line Options

```
--experiment, -e    Experiment name (partial match supported)
                    Default: current experiment from .current_experiment

--compare, -c       Compare multiple experiments by name
                    Example: --compare sparse intermediate

--output-dir, -o    Directory to save generated graphs
                    Default: experiment/analysis/graphs

--episodic          Generate episode-level graphs (loss, epsilon)

--rounds            Generate round-level win rate graphs (default)

--all               Generate all graph types

--phase             Validation phase to graph
                    Options: vs_randomized, vs_gapmaximizer, both
                    Default: both

--help              Show help message
```

## Examples

### Basic Usage
```bash
# Generate graphs for current experiment
python scripts/generate_metrics_graphs.py

# Generate all graph types
python scripts/generate_metrics_graphs.py --all

# Generate only episode-level graphs
python scripts/generate_metrics_graphs.py --episodic
```

### Working with Specific Experiments
```bash
# Generate graphs for experiment matching "sparse"
python scripts/generate_metrics_graphs.py --experiment sparse

# Compare two experiments
python scripts/generate_metrics_graphs.py --compare sparse_rewards intermediate_rewards

# Generate for specific phase only
python scripts/generate_metrics_graphs.py --phase vs_gapmaximizer
```

### Custom Output
```bash
# Save to custom directory
python scripts/generate_metrics_graphs.py --output-dir ./my_graphs

# Generate all graphs for specific experiment
python scripts/generate_metrics_graphs.py -e intermediate --all -o ./thesis_figures
```

## Output Structure

Graphs are saved to the experiment's analysis directory:
```
experiments/experiment_NAME/analysis/graphs/
├── round_win_rates.png          # Win rate by round (both phases)
├── episodic_loss.png            # Training loss over episodes
├── episodic_epsilon.png         # Exploration rate decay
└── comparison_vs_randomized.png # (when using --compare)
```

## Directory Structure

The script works with the experiment manager structure:
```
experiments/
├── experiment_NAME/
│   ├── runs/
│   │   ├── boolean_run_01/
│   │   │   ├── metrics_logs/       # Per-episode metrics (preferred)
│   │   │   │   ├── metrics_round_0_selfplay.jsonl
│   │   │   │   ├── metrics_round_0_vs_randomized_trainee_first.jsonl
│   │   │   │   └── ...
│   │   │   ├── action_logs/        # Per-turn action data
│   │   │   └── models/
│   │   └── ...
│   └── analysis/
│       └── graphs/                 # Generated graphs
└── .current_experiment             # Points to active experiment
```

## Migrating Existing Experiments

If your experiment has metrics files in `action_logs/` instead of `metrics_logs/`,
use the migration script:

```bash
# Preview what would be moved (dry-run)
python scripts/migrate_experiment_logs.py

# Actually move the files
python scripts/migrate_experiment_logs.py --execute
```

The graphing script will automatically fall back to `action_logs/` if `metrics_logs/`
doesn't exist, so migration is optional but recommended for new experiments.

## Dependencies

Required packages (included in requirements.txt):
- matplotlib: Core plotting library
- numpy: Numerical operations
- seaborn (optional): Enhanced styling

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- The script automatically combines `trainee_first` and `trainee_second` validation files
- Missing data is handled gracefully (skipped with warnings)
- Graphs are saved at 150 DPI for reasonable file sizes
- Episode-level graphs use smoothing for large datasets
- Individual runs are shown with low opacity for round-level graphs
