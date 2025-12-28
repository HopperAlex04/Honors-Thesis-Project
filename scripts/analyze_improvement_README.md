# Improvement Analysis Script

This script analyzes agent improvement by tracking win rates against fixed opponents across training rounds. This is the best way to measure whether your agents are actually learning, especially after fixing network architecture issues.

## Purpose

When training with self-play, it's hard to see improvement because both agents are learning simultaneously. By analyzing validation metrics (where agents play against fixed opponents), we can clearly see if agents are improving over time.

## Usage

### Basic Usage

```bash
# Analyze all training types against all opponents
python scripts/analyze_improvement.py

# Analyze specific training type
python scripts/analyze_improvement.py --type hand_only

# Analyze against specific opponent
python scripts/analyze_improvement.py --opponent vs_heuristic

# Combine filters
python scripts/analyze_improvement.py --type hand_only --opponent vs_gapmaximizer
```

### Options

- `--type, -t`: Training type to analyze (`hand_only`, `opponent_field_only`, `no_features`, `both_features`, or `all`)
- `--opponent, -o`: Opponent type (`vs_previous`, `vs_randomized`, `vs_heuristic`, `vs_gapmaximizer`, or `all`)
- `--base-dir, -b`: Base directory containing `action_logs` (default: current directory)
- `--output-dir, -d`: Directory to save plots (default: `improvement_analysis`)
- `--format, -f`: Output format (`png`, `svg`, or `pdf`)

## Output

The script generates:

1. **Console Summary**: Detailed statistics printed to console showing:
   - Initial vs final win rates
   - Overall improvement (final - initial)
   - Learning trend (slope of improvement)
   - R-squared value (how linear the improvement is)

2. **Visualization Plots**: Four-panel analysis showing:
   - **Win Rate Over Rounds**: Line plot showing improvement trajectory
   - **Learning Rate**: Bar chart showing improvement trend (slope)
   - **Total Improvement**: Bar chart showing overall improvement (final - initial)
   - **Final Performance**: Bar chart showing final win rates

## Interpreting Results

### Positive Indicators
- **Improvement > 0.05**: Significant improvement detected
- **Positive trend**: Agent is learning consistently
- **R² > 0.7**: Strong linear improvement pattern
- **Final rate > 0.5**: Agent beats random baseline

### Warning Signs
- **Improvement < 0**: Performance degradation
- **Trend ≈ 0**: No learning (flat line)
- **Low R²**: Erratic/unstable learning
- **Final rate < 0.5**: Worse than random

## Example Output

```
================================================================================
IMPROVEMENT ANALYSIS: VS HEURISTIC
================================================================================

Hand Only:
  Initial Win Rate: 0.450
  Final Win Rate:   0.680
  Overall Improvement: +0.230 (+23.0%)
  Learning Trend: +0.023000 per round (R² = 0.892)
  ✓ Significant improvement detected!

No Features:
  Initial Win Rate: 0.420
  Final Win Rate:   0.550
  Overall Improvement: +0.130 (+13.0%)
  Learning Trend: +0.013000 per round (R² = 0.756)
  ✓ Significant improvement detected!
```

## Use Cases

1. **After Architecture Fixes**: After removing Tanh activation, use this to verify agents are now learning properly
2. **Hyperparameter Tuning**: Compare different hyperparameter settings
3. **Feature Analysis**: Compare which feature sets lead to better learning
4. **Training Monitoring**: Track progress during long training runs

## Notes

- The script combines `trainee_first` and `trainee_second` files to get balanced statistics
- Win rates are calculated as trainee wins / total games (accounting for both player positions)
- The script automatically handles missing rounds or incomplete data

