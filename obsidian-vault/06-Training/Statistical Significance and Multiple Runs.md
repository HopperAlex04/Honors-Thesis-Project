---
title: Statistical Significance and Multiple Runs
tags: [training, experiments, statistics, research-methodology]
created: 2026-01-16
related: [Input Representation Experiments, Hyperparameters, Self-Play]
---

# Statistical Significance and Multiple Runs

## Overview

Reinforcement learning training has **high variance** due to randomness in initialization, exploration, and training dynamics. To demonstrate statistical significance in thesis research, experiments must be run **multiple times** with different random seeds, and results must be aggregated and analyzed statistically.

## Why Multiple Runs Are Essential

### The Problem with Single Runs

A single training run cannot demonstrate statistical significance because:

1. **Random Initialization**: Different weight initializations lead to different learning trajectories
2. **Stochastic Exploration**: Epsilon-greedy exploration introduces randomness in action selection
3. **Random Opponent Actions**: Even with fixed policies, randomness in game mechanics and opponent behavior
4. **Non-Deterministic Training**: Experience replay sampling, batch selection, and optimization introduce variance
5. **Path Dependence**: Early random events can significantly affect final performance

**A single run might be lucky or unlucky** - you cannot tell if results are due to the method or random chance.

### Statistical Significance Requirements

For thesis-level research, you need:

1. **Multiple independent runs** (typically 5-10 for RL experiments)
2. **Consistent experimental conditions** (same hyperparameters, same network architecture)
3. **Proper statistical analysis** (mean, standard deviation, confidence intervals)
4. **Statistical tests** (t-tests, Mann-Whitney U, etc.) to compare methods

## Recommended Number of Runs

### Standard Practice

**5-10 independent runs** is standard for RL experiments:

- **5 runs**: Minimum for basic statistical analysis
- **7-10 runs**: Better for robust statistics and confidence intervals
- **10+ runs**: Ideal for publication-quality results

### For Your Thesis

**Recommendation: 7 runs per network type**

**Rationale**:
- Enough for statistical tests (t-tests require at least 3-5, but 7 is better)
- Allows calculation of confidence intervals
- Reasonable computational cost (3 network types × 7 runs = 21 total runs)
- Standard practice in RL research

### Trade-offs

| Runs | Statistical Power | Computational Cost | Confidence |
|------|------------------|-------------------|------------|
| 3 | Low | Low | Weak |
| 5 | Moderate | Moderate | Acceptable |
| **7** | **Good** | **Moderate** | **Strong** |
| 10 | Excellent | High | Very Strong |

## Experimental Design

### Independent Runs

Each run must be **completely independent**:

- **Different random seed**: Each run uses a unique seed
- **Different weight initialization**: PyTorch random initialization
- **Different training trajectory**: Different exploration paths
- **Same hyperparameters**: Identical configuration
- **Same architecture**: Identical network structure

### What to Compare

For your input representation experiment:

- **Boolean Network**: 7 runs
- **Embedding Network**: 7 runs  
- **Multi-Encoder Network**: 7 runs

**Total**: 21 runs (3 network types × 7 runs each)

### Metrics to Track Across Runs

For each run, track:

1. **Final performance**: Win rate after training
2. **Training curves**: Win rate over episodes (for each run)
3. **Convergence speed**: Episodes to reach target performance
4. **Stability**: Variance in performance over time
5. **Final model quality**: Best checkpoint performance

## Data Management Strategy

### Directory Structure

```
experiments/
├── experiment_20260116_input_representation/
│   ├── run_001_boolean/
│   │   ├── models/
│   │   │   ├── checkpoint0.pt
│   │   │   ├── checkpoint1.pt
│   │   │   └── ...
│   │   ├── action_logs/
│   │   │   └── metrics_round_0_selfplay.jsonl
│   │   ├── metrics_logs/
│   │   │   └── metrics_round_0_selfplay.jsonl
│   │   ├── hyperparams_config.json
│   │   └── run_metadata.json
│   ├── run_002_boolean/
│   ├── run_003_embedding/
│   ├── run_004_embedding/
│   ├── ...
│   └── analysis/
│       ├── aggregated_metrics.json
│       ├── statistical_tests.json
│       └── comparison_graphs/
│           ├── win_rate_comparison.png
│           ├── training_curves_with_ci.png
│           └── ...
│
├── experiment_20260120_parameter_matching/
│   └── ...
```

### Run Identification

Each run should have:

- **Unique run ID**: `run_001`, `run_002`, etc. (zero-padded for sorting)
- **Network type**: `boolean`, `embedding`, `multi_encoder`
- **Random seed**: Stored in metadata
- **Hyperparameters**: Copy of config file
- **Timestamp**: When run started

### Metadata File

Each run should include `run_metadata.json`:

```json
{
  "run_id": "run_001",
  "network_type": "boolean",
  "random_seed": 42,
  "start_time": "2026-01-16T10:00:00",
  "end_time": "2026-01-16T12:30:00",
  "hyperparams_config": "path/to/config.json",
  "git_commit": "abc123def456",
  "total_episodes": 5000,
  "total_rounds": 10,
  "final_win_rate": 0.65,
  "final_win_rate_std": 0.08
}
```

### Experiment Metadata

Each experiment should have `experiment_metadata.json`:

```json
{
  "experiment_name": "experiment_20260116_input_representation",
  "description": "Comparison of input representation strategies",
  "network_types": ["boolean", "embedding", "multi_encoder"],
  "runs_per_type": 7,
  "total_runs": 21,
  "start_time": "2026-01-16T10:00:00",
  "end_time": "2026-01-20T18:00:00",
  "git_commit": "abc123def456",
  "hyperparams_base": "hyperparams_config.json"
}
```

## Statistical Analysis

### Aggregating Results

For each network type, aggregate across runs:

**Mean Performance**:
```
mean_win_rate = (1/n) * Σ(win_rate_i)
```

**Standard Deviation**:
```
std_win_rate = sqrt((1/(n-1)) * Σ(win_rate_i - mean_win_rate)²)
```

**95% Confidence Interval**:
```
CI = mean ± t_(α/2, n-1) * (std / sqrt(n))
```

Where:
- `n` = number of runs (7)
- `t_(α/2, n-1)` = t-statistic for 95% CI with n-1 degrees of freedom
- For n=7: t ≈ 2.447

### Comparing Networks

Use statistical tests to compare network types:

#### 1. Normality Test

**Shapiro-Wilk test**: Check if data is normally distributed
- If p > 0.05: Use parametric tests (t-test)
- If p ≤ 0.05: Use non-parametric tests (Mann-Whitney U)

#### 2. Equal Variance Test

**Levene's test**: Check if variances are equal
- If p > 0.05: Variances are equal (use standard t-test)
- If p ≤ 0.05: Variances are unequal (use Welch's t-test)

#### 3. Mean Comparison

**Independent t-test** (if normal):
- Compares means of two groups
- Null hypothesis: Means are equal
- Alternative: Means are different

**Mann-Whitney U test** (if not normal):
- Non-parametric alternative
- Compares distributions
- More robust to outliers

#### 4. Multiple Comparisons

If comparing 3 network types, use **Bonferroni correction**:
- Adjust significance level: α_adjusted = α / number_of_comparisons
- For 3 comparisons: α_adjusted = 0.05 / 3 = 0.0167

### Visualization

Create plots with:

1. **Mean line**: Average across runs
2. **Confidence bands**: 95% CI shaded region
3. **Individual runs**: Light lines showing variance
4. **Statistical annotations**: p-values, significance markers (*, **, ***)

**Example**:
```
Win Rate Comparison
==================
Boolean:     0.65 ± 0.08 (95% CI: [0.58, 0.72])
Embedding:   0.72 ± 0.06 (95% CI: [0.67, 0.77])
Multi-Enc:   0.70 ± 0.09 (95% CI: [0.62, 0.78])

Statistical Tests:
- Boolean vs Embedding: p = 0.023* (t-test)
- Boolean vs Multi-Enc: p = 0.156 (t-test)
- Embedding vs Multi-Enc: p = 0.412 (t-test)
```

## Implementation Plan

### Phase 1: Single Run (Current)

✅ You have: Single run training script
✅ You have: Metrics logging
✅ You have: Checkpoint saving

### Phase 2: Multiple Runs (To Implement)

**Need to add**:

1. **Run management script**: Automates multiple runs
   - Generates unique seeds
   - Creates run directories
   - Manages parallel execution
   - Tracks run status

2. **Run metadata tracking**: Records run information
   - Random seed
   - Start/end time
   - Git commit
   - Configuration

3. **Directory organization**: Structured storage
   - Experiment-level directories
   - Run-level subdirectories
   - Analysis directories

4. **Seed management**: Reproducible randomness
   - Set PyTorch random seed
   - Set Python random seed
   - Set NumPy random seed
   - Set environment seed (if applicable)

### Phase 3: Analysis (To Implement)

**Need to add**:

1. **Aggregation script**: Combines metrics across runs
   - Loads metrics from all runs
   - Calculates mean, std, CI
   - Saves aggregated data

2. **Statistical analysis script**: Performs tests
   - Normality tests
   - Variance tests
   - Mean comparison tests
   - Multiple comparison correction

3. **Comparison visualization**: Creates plots
   - Training curves with CI
   - Win rate comparisons
   - Statistical annotations

4. **Report generation**: Creates summary
   - Mean performance table
   - Statistical test results
   - Visualization summary

## Best Practices

### 1. Consistent Conditions

- **Same hyperparameters** across all runs
- **Same training schedule** (rounds, episodes per round)
- **Same validation opponents**
- **Only vary**: random seed, initialization

### 2. Reproducibility

- **Save random seed** for each run
- **Save git commit hash** (code version)
- **Save exact hyperparameters** (config file)
- **Document any manual interventions**

### 3. Early Stopping

- **Use same early stopping criteria** for all runs
- **Document if/when early stopping triggered**
- **Include stopped runs in analysis** (with note)

### 4. Resource Management

- **Run sequentially or in parallel** (if resources allow)
- **Monitor resource usage** (CPU, memory, disk)
- **Save intermediate checkpoints** (for long runs)
- **Archive completed runs** (to save space)

### 5. Data Integrity

- **Verify all runs completed** before analysis
- **Check for corrupted files** (metrics, checkpoints)
- **Validate metrics format** (consistent structure)
- **Backup important runs** (before deletion)

## Computational Considerations

### Time Estimate

Assuming each run takes ~4 hours:

- **7 runs per network**: 28 hours
- **3 network types**: 84 hours total (~3.5 days)
- **Sequential execution**: 3.5 days wall-clock time
- **Parallel execution** (3 runs): ~1.2 days wall-clock time

### Parallel Execution

If you have multiple GPUs/CPUs:

- Can run multiple runs in parallel
- Reduces wall-clock time
- Requires careful resource management
- Need to ensure no resource conflicts

**Recommendation**: Start with sequential, then parallel if needed

### Storage

Each run generates:

- **Models**: ~10-50 MB per checkpoint
- **Logs**: ~100-500 MB per run (metrics, actions)
- **Total per run**: ~1-2 GB
- **21 runs**: ~20-40 GB total

**Recommendation**: Archive old runs, keep only final models

## Thesis Presentation

### Results Section

Present:

1. **Mean performance** with confidence intervals
   - "Boolean network achieved 0.65 ± 0.08 win rate (95% CI: [0.58, 0.72])"

2. **Statistical tests** comparing networks
   - "Embedding network significantly outperformed boolean (p = 0.023, t-test)"

3. **Training curves** showing mean ± CI
   - Figure with shaded confidence bands

4. **Variance analysis** (which is more stable?)
   - "Multi-encoder network showed higher variance (std = 0.09 vs 0.06)"

### Discussion

Address:

- **Why some networks are more variable**: Architecture, training dynamics
- **Whether differences are statistically significant**: Statistical test results
- **Practical significance vs statistical significance**: Effect sizes
- **Limitations of the study**: Sample size, assumptions

### Figures

Include:

1. **Training curves**: Mean ± 95% CI across runs
2. **Final performance comparison**: Bar plot with error bars
3. **Statistical test results**: Table with p-values
4. **Individual run trajectories**: Light lines showing variance

## Related Concepts

- [[Input Representation Experiments]] - Experimental design
- [[Hyperparameters]] - Configuration management
- [[Self-Play]] - Training methodology
- [[Deep Q-Network]] - Algorithm being tested

## Further Reading

- Henderson, P., et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI*.
- Colas, C., et al. (2019). "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments." *arXiv*.

---
*Guidelines for running multiple experiments and demonstrating statistical significance*
