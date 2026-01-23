# Run Count Reduction: Statistical Significance Analysis

## Decision

**Reduced runs per network type from 7 to 5** to save time while maintaining statistical significance.

## Statistical Significance Comparison

### 5 Runs (Selected)

**Statistical Power**: Moderate to Good
- ✅ **Minimum for basic statistical analysis** (per RL research standards)
- ✅ Allows t-tests and confidence intervals
- ✅ Sufficient for thesis-level research
- ⚠️ Slightly wider confidence intervals than 7 runs

**95% Confidence Interval**:
- t-statistic for n=5: **t₀.₀₂₅,₄ = 2.776**
- Standard Error multiplier: 2.776 / √5 = **1.241**
- Example: If mean = 0.70, std = 0.08 → CI = 0.70 ± 0.099

**Time Savings**:
- Total runs: 15 (down from 21)
- **29% reduction** in total experiment time
- Saves ~6 runs worth of time

### 7 Runs (Previous)

**Statistical Power**: Good to Strong
- ✅ Better for robust statistics
- ✅ Narrower confidence intervals
- ✅ More standard in RL research
- ❌ Higher computational cost

**95% Confidence Interval**:
- t-statistic for n=7: **t₀.₀₂₅,₆ = 2.447**
- Standard Error multiplier: 2.447 / √7 = **0.925**
- Example: If mean = 0.70, std = 0.08 → CI = 0.70 ± 0.074

### Comparison

| Metric | 5 Runs | 7 Runs | Difference |
|--------|--------|--------|------------|
| t-statistic | 2.776 | 2.447 | +13% wider CI |
| CI width (example) | ±0.099 | ±0.074 | +34% wider |
| Statistical power | Moderate | Good | Acceptable |
| Total runs | 15 | 21 | -29% time |
| Thesis adequacy | ✅ Yes | ✅ Yes | Both acceptable |

## Justification for 5 Runs

### 1. Minimum Standard Met

Per documentation and RL research standards:
- **5 runs is the minimum** for basic statistical analysis
- Still allows proper statistical tests (t-tests, confidence intervals)
- Acceptable for undergraduate thesis research

### 2. Time Constraint

- **29% time savings** (6 fewer runs)
- Each run takes ~1.5-2 hours
- Saves ~9-12 hours of total experiment time
- Critical given time pressure

### 3. Validation Quality Maintained

- Kept `validation_episodes_ratio = 1.0` (250 validation per round)
- Maintains high-quality validation estimates
- Better to have fewer runs with better validation than more runs with less validation

### 4. Statistical Tests Still Valid

**With 5 runs per type**:
- ✅ Can perform t-tests (minimum n=3-5, 5 is acceptable)
- ✅ Can calculate 95% confidence intervals
- ✅ Can perform pairwise comparisons
- ✅ Can use Bonferroni correction for multiple comparisons
- ⚠️ Slightly less power to detect small differences

**Example Statistical Analysis**:
```
Network Type Comparison (5 runs each):
========================================
Linear:        0.XX ± 0.XX (95% CI: [0.XX, 0.XX])
Large Hidden:  0.XX ± 0.XX (95% CI: [0.XX, 0.XX])
Game-Based:    0.XX ± 0.XX (95% CI: [0.XX, 0.XX])

Pairwise t-tests (Bonferroni corrected, α = 0.0167):
- Linear vs Large Hidden: p = X.XXX
- Linear vs Game-Based:   p = X.XXX
- Large Hidden vs Game-Based: p = X.XXX
```

## Thesis Presentation

### How to Present in Thesis

**Methods Section**:
> "Each network architecture was trained 5 times with different random seeds to ensure statistical validity. This sample size allows for proper statistical analysis including t-tests and 95% confidence intervals, while balancing computational constraints."

**Results Section**:
> "Results are reported as mean ± standard deviation with 95% confidence intervals. Statistical comparisons between architectures were performed using independent t-tests with Bonferroni correction for multiple comparisons (α = 0.0167)."

**Limitations Section** (if needed):
> "The study used 5 runs per architecture type, which is the minimum recommended for statistical analysis in reinforcement learning. While this provides sufficient power for detecting meaningful differences, a larger sample size (7-10 runs) would provide narrower confidence intervals and greater statistical power."

## Updated Configuration

```json
{
  "training": {
    "rounds": 20,
    "eps_per_round": 250,
    "validation_episodes_ratio": 1.0
  },
  "eps_decay": 11000
}
```

**Experiment Setup**:
- **Runs per type**: 5 (reduced from 7)
- **Total runs**: 15 (3 types × 5 runs)
- **Validation**: 250 episodes per round (maintained)
- **Total episodes per run**: 10,000 (5,000 training + 5,000 validation)

## Time Savings Summary

| Configuration | Runs | Time per Run | Total Time |
|--------------|------|--------------|------------|
| Previous (7 runs) | 21 | ~1.5-2h | ~31.5-42h |
| **New (5 runs)** | **15** | **~1.5-2h** | **~22.5-30h** |
| **Savings** | **-6** | - | **~9-12h** |

**Savings**: ~29% reduction in total experiment time while maintaining statistical validity.
