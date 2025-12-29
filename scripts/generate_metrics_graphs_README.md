# Metrics Graph Generation Script

## Quick Start

Generate graphs for all training types and phases:
```bash
python scripts/generate_metrics_graphs.py
```

Generate graphs for a specific training type:
```bash
python scripts/generate_metrics_graphs.py --type hand_only
```

Generate only win rate graphs:
```bash
python scripts/generate_metrics_graphs.py --metrics win_rate
```

Generate comparison graphs across all training types:
```bash
python scripts/generate_metrics_graphs.py --comparisons
```

## Features

### Primary Features
- Parse JSONL metrics files from organized subdirectories
- Aggregate episodes by training phase across rounds
- Generate visualization graphs with matplotlib/seaborn
- Support all training types (hand_only, opponent_field_only, no_features, both_features, all_features, scores)
- Support all training phases (selfplay, vs_previous, vs_randomized, vs_heuristic, vs_gapmaximizer)
- Combine trainee_first and trainee_second for validation phases

### Statistical Analysis
- Mean, standard deviation, min, max, median
- Trend analysis (linear regression slope)
- R-squared for trend quality
- Error bars on graphs (when std dev available)
- Statistics displayed in text box on graphs

### Graph Types
- Win rate trends (p1_win_rate, p2_win_rate, draw_rate)
- Training loss (only during selfplay)
- Epsilon decay (exploration vs exploitation)
- Memory size growth (replay buffer size)
- Score distributions
- Episode length (turns per episode)
- Comparison graphs across training types

## Command-Line Options

```
--type, -t          Training type to analyze (hand_only, opponent_field_only, 
                     no_features, both_features, all_features, scores). 
                     Default: all types.

--phase, -p          Training phase(s) to visualize (comma-separated).
                     Options: selfplay, vs_previous, vs_randomized, 
                     vs_heuristic, vs_gapmaximizer. Default: all phases.

--output-dir, -o     Directory to save generated graphs.
                     Default: ./metrics_graphs

--format, -f         Output format: png, svg, pdf
                     Default: png

--metrics, -m        Comma-separated list of metrics to plot.
                     Options: win_rate, loss, epsilon, memory_size, score, turns
                     Or specific metric names like p1_win_rate, loss, etc.
                     Default: all common metrics

--rounds             Comma-separated list of rounds to include (e.g., "0,1,2,3")
                     Default: all available rounds

--style, -s          Plot style: default, seaborn, ggplot
                     Default: seaborn

--no-combine         Don't combine trainee_first and trainee_second for validation phases

--base-dir            Base directory containing action_logs
                     Default: current directory

--comparisons         Generate comparison graphs across training types

--help               Show help message
```

## Examples

### Basic Usage
```bash
# Generate all graphs for hand_only training
python scripts/generate_metrics_graphs.py --type hand_only

# Generate only self-play graphs for all training types
python scripts/generate_metrics_graphs.py --phase selfplay

# Generate win rate graphs only
python scripts/generate_metrics_graphs.py --metrics win_rate

# Generate graphs for specific rounds
python scripts/generate_metrics_graphs.py --rounds 0,1,2,3,4,5
```

### Advanced Usage
```bash
# Generate comparison graphs across all training types
python scripts/generate_metrics_graphs.py --phase vs_gapmaximizer --comparisons

# Generate high-quality PDF graphs
python scripts/generate_metrics_graphs.py --format pdf --style seaborn

# Generate only validation phase graphs
python scripts/generate_metrics_graphs.py --phase vs_previous,vs_randomized,vs_heuristic,vs_gapmaximizer

# Generate specific metrics for specific training type
python scripts/generate_metrics_graphs.py --type hand_only --metrics win_rate,loss,epsilon --phase selfplay
```

## Output Structure

Graphs are organized in the following structure:
```
metrics_graphs/
├── hand_only/
│   ├── selfplay/
│   │   ├── p1_win_rate_selfplay_hand_only.png
│   │   ├── loss_selfplay_hand_only.png
│   │   └── ...
│   ├── vs_previous/
│   │   ├── p1_win_rate_vs_previous_hand_only.png
│   │   └── ...
│   └── ...
├── opponent_field_only/
├── no_features/
├── both_features/  (legacy)
├── all_features/
├── scores/
└── comparisons/
    ├── p1_win_rate_comparison_vs_gapmaximizer.png
    └── ...
```

## Statistical Analysis

Each graph includes a statistics box showing:
- **Mean**: Average value across all rounds
- **Std**: Standard deviation
- **R²**: R-squared value for trend quality (0-1, higher is better)
- **Trend**: Linear regression slope (positive = improving, negative = degrading)

Error bars on graphs show standard deviation when available.

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_generate_metrics_graphs.py -v

# Run specific test class
python -m pytest tests/test_generate_metrics_graphs.py::TestParseMetricsFile -v

# Run with coverage
python -m pytest tests/test_generate_metrics_graphs.py --cov=scripts/generate_metrics_graphs
```

## Dependencies

Required packages (included in requirements.txt):
- matplotlib: Core plotting library
- seaborn: Enhanced styling and statistical plots
- numpy: Numerical operations for aggregations
- pytest: Testing framework (for running tests)

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- The script automatically combines `trainee_first` and `trainee_second` files for validation phases
- Missing rounds or files are handled gracefully (skipped with warnings)
- NaN values are filtered out during aggregation
- Graphs are saved at 300 DPI for high quality
- Statistics are calculated only on valid (non-NaN) values

