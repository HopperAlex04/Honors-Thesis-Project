---
title: Experiment Management System
tags: [training, experiments, automation, tools, implementation]
created: 2026-01-19
related: [Statistical Significance and Multiple Runs, Input Representation Experiments, Training Time Estimates]
---

# Experiment Management System

## Overview

The Experiment Management System is a suite of Python scripts that automates running the full input representation comparison experiment (3 network types × 7 runs = 21 total runs). It handles experiment initialization, run execution, status tracking, and statistical analysis.

**Status**: ✅ **Implemented** (January 2026)

## System Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment Management System                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  experiment_     │   │  run_full_       │   │  aggregate_  │ │
│  │  manager.py      │   │  experiment.py   │   │  experiment_ │ │
│  │                  │   │                  │   │  results.py  │ │
│  │  • Initialize    │   │  • Sequential    │   │              │ │
│  │  • Track status  │──▶│    execution     │──▶│  • Aggregate │ │
│  │  • Manage runs   │   │  • Parallel      │   │  • Analyze   │ │
│  │  • Store config  │   │    execution     │   │  • Visualize │ │
│  └──────────────────┘   └──────────────────┘   └──────────────┘ │
│           │                      │                     │         │
│           ▼                      ▼                     ▼         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    experiments/ directory                    ││
│  │  ├── experiment_YYYYMMDD_name/                              ││
│  │  │   ├── runs/                                              ││
│  │  │   │   ├── boolean_run_01/                                ││
│  │  │   │   ├── boolean_run_02/                                ││
│  │  │   │   └── ...                                            ││
│  │  │   ├── analysis/                                          ││
│  │  │   ├── experiment_metadata.json                           ││
│  │  │   ├── runs_status.json                                   ││
│  │  │   └── base_hyperparams_config.json                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Scripts

### 1. experiment_manager.py

**Purpose**: Core experiment management - initialize, track, and organize experiments

**Location**: `scripts/experiment_manager.py`

**Commands**:

| Command | Description |
|---------|-------------|
| `init --name NAME` | Initialize a new experiment |
| `status` | Show current experiment status |
| `list` | List all experiments |
| `select NAME` | Select an experiment as current |
| `run --type TYPE --run N` | Execute a single specific run |
| `analyze` | Analyze completed runs |
| `export --format FMT` | Export data (csv, json) |

**Key Functions**:
- `ExperimentManager.init_experiment()` - Creates experiment directory structure
- `ExperimentManager.prepare_run()` - Sets up individual run configuration
- `ExperimentManager.complete_run()` - Records run completion and metrics
- `generate_seeds()` - Creates deterministic seeds for reproducibility

### 2. run_full_experiment.py

**Purpose**: Execute training runs, either sequentially or in parallel

**Location**: `scripts/run_full_experiment.py`

**Options**:

| Option | Description |
|--------|-------------|
| `--parallel N` | Run N sessions in parallel (default: 1) |
| `--type TYPE` | Only run specific network type |
| `--runs 1,2,3` | Only run specific run numbers |
| `--dry-run` | Preview without executing |
| `--force` | Re-run completed runs |
| `--init NAME` | Initialize before running |

**Execution Modes**:
- **Sequential**: Runs one training at a time (safer, less resource intensive)
- **Parallel**: Runs multiple trainings simultaneously (faster, more resource intensive)

### 3. aggregate_experiment_results.py

**Purpose**: Analyze results and generate visualizations

**Location**: `scripts/aggregate_experiment_results.py`

**Options**:

| Option | Description |
|--------|-------------|
| `--graphs` | Generate comparison visualizations |
| `--export FORMAT` | Export results (json, csv, latex) |
| `--output-dir PATH` | Custom output directory |

**Output**:
- Statistical summaries (mean, std, CI)
- Pairwise t-tests with Bonferroni correction
- ANOVA across all groups
- Effect sizes (Cohen's d, η²)

## How the System Works

### Phase 1: Initialization

When you run `experiment_manager.py init`:

1. **Creates directory structure**:
   ```
   experiments/experiment_YYYYMMDD_HHMMSS_name/
   ├── runs/           # Will contain 21 run subdirectories
   ├── analysis/       # Will contain results and graphs
   │   └── graphs/
   ├── experiment_metadata.json
   ├── runs_status.json
   └── base_hyperparams_config.json
   ```

2. **Generates 21 unique seeds** deterministically (for reproducibility)

3. **Creates run tracking** for all 21 runs:
   - `boolean_run_01` through `boolean_run_07`
   - `embedding_run_01` through `embedding_run_07`
   - `multi_encoder_run_01` through `multi_encoder_run_07`

4. **Copies base configuration** from `hyperparams_config.json`

### Phase 2: Execution

When you run `run_full_experiment.py`:

1. **For each pending run**:
   - Creates run directory with its own config
   - Sets `network_type` and `random_seed` in config
   - Creates isolated workspace (models/, action_logs/)
   - Executes `train.py` with run-specific settings
   - Collects results and metrics
   - Updates run status

2. **Tracking**:
   - Status: `pending` → `prepared` → `running` → `completed`/`failed`
   - Records start time, end time, duration
   - Stores final metrics (win rate, episodes, etc.)

3. **Resumability**:
   - If interrupted, running again skips completed runs
   - Can use `--force` to re-run completed runs

### Phase 3: Analysis

When you run `aggregate_experiment_results.py`:

1. **Collects metrics** from all completed runs

2. **Calculates statistics** per network type:
   - Mean win rate
   - Standard deviation
   - Standard error
   - 95% confidence interval
   - Min, max, median

3. **Performs statistical tests**:
   - **Normality check**: Shapiro-Wilk test
   - **Variance check**: Levene's test
   - **Pairwise comparisons**: t-test or Mann-Whitney U
   - **Multiple comparison correction**: Bonferroni
   - **Overall comparison**: One-way ANOVA
   - **Effect sizes**: Cohen's d, η²

4. **Generates visualizations**:
   - Bar chart with 95% CI error bars
   - Training curves with confidence bands
   - Box plots with individual data points

## Directory Structure

After running all experiments:

```
experiments/
└── experiment_20260119_input_rep_v1/
    ├── runs/
    │   ├── boolean_run_01/
    │   │   ├── models/
    │   │   │   ├── model_initial.pt    # Before training
    │   │   │   └── model_final.pt      # After training
    │   │   ├── action_logs/
    │   │   │   ├── metrics_round_0_selfplay.jsonl
    │   │   │   └── ...
    │   │   ├── hyperparams_config.json   # Run-specific config
    │   │   └── run_metadata.json         # Run metadata
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

**Note**: Each run saves only initial and final models (no intermediate checkpoints). If interrupted, a run restarts from the beginning. With ~1 hour per run, this is acceptable.

## Reproducibility

### Random Seed Management

The experiment system ensures reproducibility through deterministic random seed handling:

1. **Seed Generation**: When an experiment is initialized, 21 unique seeds are generated from a base seed (42) using `numpy.random.seed(base_seed)`. This ensures the same seeds are generated each time an experiment is initialized with the same base seed.

2. **Seed Application**: Each run has a unique seed stored in its configuration. The `train.py` script applies this seed to all random number generators:
   - Python's `random` module
   - NumPy's `np.random`
   - PyTorch's `torch.manual_seed`
   - CUDA seeds (if GPU is available)

3. **Seed Storage**: Seeds are saved in:
   - `experiment_metadata.json`: All 21 seeds for the experiment
   - `run_metadata.json`: Individual run's seed
   - `model_initial.pt` and `model_final.pt`: Seed used for training

### Network Initialization

Networks use Kaiming/He initialization, which is the recommended practice for ReLU networks:
- Prevents vanishing/exploding gradients
- Ensures proper signal propagation
- Initialized deterministically when random seed is set

### Environment Reproducibility

The Cuttle environment's `reset()` method accepts a seed parameter, ensuring deterministic:
- Card shuffling
- Deck draws
- Any randomized game mechanics

## Statistical Methods

### Descriptive Statistics

For each network type with n=7 runs:

| Statistic | Formula | Purpose |
|-----------|---------|---------|
| Mean | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | Central tendency |
| Std Dev | $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$ | Variability |
| Std Error | $SE = \frac{s}{\sqrt{n}}$ | Precision of mean |
| 95% CI | $\bar{x} \pm t_{0.975, n-1} \cdot SE$ | Confidence interval |

### Inferential Statistics

**Pairwise Comparisons**:

1. **Normality Test** (Shapiro-Wilk):
   - H₀: Data is normally distributed
   - If p > 0.05: Use parametric tests
   - If p ≤ 0.05: Use non-parametric tests

2. **Variance Test** (Levene's):
   - H₀: Variances are equal
   - If p > 0.05: Use standard t-test
   - If p ≤ 0.05: Use Welch's t-test

3. **Mean Comparison**:
   - **t-test**: If normal, equal variance
   - **Welch's t-test**: If normal, unequal variance
   - **Mann-Whitney U**: If non-normal

4. **Multiple Comparison Correction** (Bonferroni):
   - Adjusted α = 0.05 / 3 = 0.0167
   - Prevents false positives from multiple tests

**Overall Comparison** (One-way ANOVA):
- Compares all three groups simultaneously
- F-statistic indicates overall difference
- Effect size: η² (eta-squared)

### Effect Sizes

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

## Usage Examples

### Complete Workflow

```bash
# 1. Initialize experiment
python scripts/experiment_manager.py init --name "input_rep_v1" \
    --description "Input representation comparison"

# 2. Check status
python scripts/experiment_manager.py status

# 3. Run all experiments (sequential)
python scripts/run_full_experiment.py

# OR run with parallelization
python scripts/run_full_experiment.py --parallel 3

# 4. Monitor progress
python scripts/experiment_manager.py status

# 5. Analyze results
python scripts/aggregate_experiment_results.py --graphs --export latex
```

### Running Specific Subsets

```bash
# Run only boolean network experiments
python scripts/run_full_experiment.py --type boolean

# Run only runs 1, 2, 3 for all types
python scripts/run_full_experiment.py --runs 1,2,3

# Run a single specific run
python scripts/experiment_manager.py run --type embedding --run 5
```

### Managing Multiple Experiments

```bash
# List all experiments
python scripts/experiment_manager.py list

# Switch to a different experiment
python scripts/experiment_manager.py select input_rep_v1
```

## Configuration

### Base Hyperparameters

Before initializing an experiment, edit `hyperparams_config.json`:

```json
{
  "network_type": "boolean",  // Will be overridden per run
  "training": {
    "rounds": 20,
    "eps_per_round": 250,
    "quick_test_mode": false,  // Should be false for real experiments
    "validation_episodes_ratio": 0.5,
    "validation_opponent": "both"
  }
}
```

### Experiment Constants

Configurable in `experiment_manager.py`:

```python
NETWORK_TYPES = ["boolean", "embedding", "multi_encoder"]
RUNS_PER_TYPE = 7
TOTAL_RUNS = 21
```

## Time and Storage Estimates

### Time Estimates

Based on measured training times (~1 hour per run):

| Scenario | Per Run | All 21 (Sequential) | Parallel (3) |
|----------|---------|---------------------|--------------|
| Optimistic | 45 min | 16 hours | 5 hours |
| Average | 1 hour | 21 hours | 7 hours |
| Pessimistic | 1.5 hours | 32 hours | 11 hours |

### Storage Estimates

| Component | Per Run | All 21 Runs |
|-----------|---------|-------------|
| Models | 1-2 GB | 21-42 GB |
| Logs | 0.5-1 GB | 10-21 GB |
| **Total** | 1.5-3 GB | 31-63 GB |

## Troubleshooting

### Run Failed

```bash
# Check error message
python scripts/experiment_manager.py status

# Re-run the failed run
python scripts/experiment_manager.py run --type TYPE --run N --force
```

### Missing Dependencies

```bash
# Install required packages
pip install scipy pandas matplotlib seaborn
```

### Disk Space Issues

```bash
# Keep only final checkpoints
find experiments/*/runs/*/models -name "checkpoint*.pt" \
    ! -name "checkpoint20.pt" -delete
```

### Resume After Interruption

```bash
# Simply run again - completed runs are skipped
python scripts/run_full_experiment.py
```

## Generated Visualizations

### 1. Win Rate Comparison (Bar Chart)

- Bar height = mean win rate
- Error bars = 95% confidence interval
- Individual points = each run's result
- Suitable for: Final results presentation

### 2. Training Curves

- Lines = mean win rate per round
- Shaded bands = ± 1 standard deviation
- Light lines = individual runs
- Suitable for: Learning dynamics comparison

### 3. Win Rate Distribution (Box Plot)

- Box = interquartile range
- Whiskers = min/max (excluding outliers)
- Points = individual runs
- Suitable for: Distribution comparison

## Output Formats

### JSON Export

```json
{
  "generated_at": "2026-01-19T12:00:00",
  "aggregated_results": {
    "boolean": {
      "n_runs": 7,
      "mean": 0.65,
      "std": 0.08,
      "ci_95_lower": 0.58,
      "ci_95_upper": 0.72
    }
  },
  "statistical_tests": {
    "pairwise_ttests": {...},
    "anova": {...}
  }
}
```

### LaTeX Export

```latex
\begin{table}[h]
\centering
\caption{Win Rate by Network Type}
\begin{tabular}{lcccc}
\hline
Network Type & N & Mean & Std & 95\% CI \\
\hline
Boolean & 7 & 65.0\% & 8.0\% & [58.0\%, 72.0\%] \\
...
\end{tabular}
\end{table}
```

## Related Concepts

- [[Statistical Significance and Multiple Runs]] - Why multiple runs are needed
- [[Input Representation Experiments]] - Experimental design
- [[Training Time Estimates]] - Time planning
- [[Hyperparameters]] - Configuration settings

## References

- Henderson, P., et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI*.
- Colas, C., et al. (2019). "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments."

---
*Documentation for the experiment management system*
