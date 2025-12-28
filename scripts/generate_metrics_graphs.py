#!/usr/bin/env python3
"""
Script to generate visualization graphs from training metrics files.

This script aggregates metrics across training rounds and generates comprehensive
visualizations to track training progress.

Usage:
    python scripts/generate_metrics_graphs.py --type hand_only
    python scripts/generate_metrics_graphs.py --phase selfplay --metrics win_rate
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Valid training types
VALID_TRAINING_TYPES = [
    "hand_only",
    "opponent_field_only",
    "no_features",
    "both_features"
]

# Valid phases
VALID_PHASES = [
    "selfplay",
    "vs_previous",
    "vs_randomized",
    "vs_heuristic",
    "vs_gapmaximizer"
]

# Metric groups
METRIC_GROUPS = {
    "win_rate": ["p1_win_rate", "p2_win_rate", "draw_rate"],
    "loss": ["loss"],
    "epsilon": ["p1_epsilon", "p2_epsilon"],
    "memory_size": ["p1_memory_size", "p2_memory_size"],
    "score": ["p1_score", "p2_score"],
    "turns": ["episode_turns"]
}


def parse_metrics_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a JSONL metrics file and return list of episode metrics.
    
    Args:
        file_path: Path to the metrics JSONL file
        
    Returns:
        List of episode metric dictionaries
    """
    metrics = []
    if not file_path.exists():
        return metrics
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON in {file_path} at line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return metrics


def extract_round_number(filename: str) -> Optional[int]:
    """
    Extract round number from metrics filename.
    
    Args:
        filename: Metrics filename (e.g., "metrics_hand_only_round_5_selfplay.jsonl")
        
    Returns:
        Round number or None if not found
    """
    match = re.search(r'_round_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def aggregate_phase_metrics(
    training_type: str,
    phase: str,
    base_dir: Path,
    combine_positions: bool = True,
    rounds_filter: Optional[List[int]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Aggregate metrics for a specific phase across all rounds.
    
    For validation phases, combines trainee_first and trainee_second files.
    
    Args:
        training_type: Training type (e.g., "hand_only")
        phase: Phase name (e.g., "selfplay", "vs_previous")
        base_dir: Base directory containing action_logs
        combine_positions: If True, combine trainee_first and trainee_second
        rounds_filter: Optional list of round numbers to include
        
    Returns:
        Dictionary mapping round number to list of episode metrics
    """
    metrics_by_round = {}
    log_dir = base_dir / "action_logs" / training_type
    
    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        return metrics_by_round
    
    if phase == "selfplay":
        # Single file per round
        pattern = f"metrics_{training_type}_round_*_selfplay.jsonl"
        for file_path in log_dir.glob(pattern):
            round_num = extract_round_number(file_path.name)
            if round_num is None:
                continue
            if rounds_filter is not None and round_num not in rounds_filter:
                continue
            
            episodes = parse_metrics_file(file_path)
            if episodes:
                metrics_by_round[round_num] = episodes
    else:
        # Two files per round (trainee_first and trainee_second)
        # Phase already includes "vs_" prefix, so don't add it again
        first_pattern = f"metrics_{training_type}_round_*_{phase}_trainee_first.jsonl"
        second_pattern = f"metrics_{training_type}_round_*_{phase}_trainee_second.jsonl"
        
        # Filter out None values from extract_round_number
        first_files = {}
        for f in log_dir.glob(first_pattern):
            round_num = extract_round_number(f.name)
            if round_num is not None:
                first_files[round_num] = f
        
        second_files = {}
        for f in log_dir.glob(second_pattern):
            round_num = extract_round_number(f.name)
            if round_num is not None:
                second_files[round_num] = f
        
        all_rounds = set(first_files.keys()) | set(second_files.keys())
        
        for round_num in all_rounds:
            if rounds_filter is not None and round_num not in rounds_filter:
                continue
            
            episodes = []
            
            if round_num in first_files:
                file_path = first_files[round_num]
                first_episodes = parse_metrics_file(file_path)
                episodes.extend(first_episodes)
            
            if combine_positions and round_num in second_files:
                file_path = second_files[round_num]
                second_episodes = parse_metrics_file(file_path)
                episodes.extend(second_episodes)
            
            # Only add if we have episodes
            if episodes:
                metrics_by_round[round_num] = episodes
    
    return metrics_by_round


def extract_metric_series(
    metrics_by_round: Dict[int, List[Dict[str, Any]]],
    metric_name: str,
    aggregation: str = "mean"
) -> Tuple[List[int], List[float], Optional[List[float]]]:
    """
    Extract a metric series across episodes (not aggregated by round).
    
    Args:
        metrics_by_round: Dictionary mapping round to episode metrics
        metric_name: Name of metric to extract
        aggregation: Ignored (kept for backward compatibility)
        
    Returns:
        Tuple of (episode_numbers, metric_values, None)
        Episode numbers are cumulative across rounds (episode 0 from round 0, episode 0 from round 1, etc.)
    """
    episode_numbers = []
    values = []
    
    # Sort rounds to process in order
    rounds = sorted(metrics_by_round.keys())
    cumulative_episode = 0
    
    for round_num in rounds:
        episodes = metrics_by_round[round_num]
        # Sort episodes by episode number if available
        sorted_episodes = sorted(episodes, key=lambda ep: ep.get("episode", 0))
        
        for ep in sorted_episodes:
            if metric_name in ep and ep[metric_name] is not None:
                # Use cumulative episode count to ensure unique x-axis values
                # This handles cases where episodes restart at 0 for each round
                episode_numbers.append(cumulative_episode)
                values.append(float(ep[metric_name]))
                cumulative_episode += 1
    
    # Return None for std_devs since we're not aggregating
    return episode_numbers, values, None


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for a metric series.
    
    Args:
        values: List of metric values
        
    Returns:
        Dictionary with statistical measures
    """
    valid_values = [v for v in values if not np.isnan(v)]
    
    if not valid_values:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "trend": np.nan,
            "r_squared": np.nan
        }
    
    valid_values = np.array(valid_values)
    
    # Calculate basic statistics
    stats = {
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "median": float(np.median(valid_values))
    }
    
    # Calculate trend (linear regression slope)
    if len(valid_values) > 1:
        x = np.arange(len(valid_values))
        coeffs = np.polyfit(x, valid_values, 1)
        stats["trend"] = float(coeffs[0])  # Slope
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((valid_values - y_pred) ** 2)
        ss_tot = np.sum((valid_values - np.mean(valid_values)) ** 2)
        if ss_tot > 0:
            stats["r_squared"] = float(1 - (ss_res / ss_tot))
        else:
            stats["r_squared"] = 0.0
    else:
        stats["trend"] = 0.0
        stats["r_squared"] = 0.0
    
    return stats


def compute_moving_average(values: List[float], window_size: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute moving average with standard deviation for confidence intervals.
    
    Args:
        values: List of metric values
        window_size: Size of the moving window
        
    Returns:
        Tuple of (smoothed_values, lower_bounds, upper_bounds)
    """
    if len(values) < window_size:
        return values, values, values
    
    smoothed = []
    lower_bounds = []
    upper_bounds = []
    
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        window_values = [v for v in values[start:end] if not np.isnan(v)]
        
        if window_values:
            mean_val = np.mean(window_values)
            std_val = np.std(window_values)
            smoothed.append(mean_val)
            lower_bounds.append(mean_val - std_val)
            upper_bounds.append(mean_val + std_val)
        else:
            smoothed.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    
    return smoothed, lower_bounds, upper_bounds


def bin_episodes(
    episodes: List[int],
    values: List[float],
    bin_size: int
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Bin episodes into groups and compute mean and std for each bin.
    
    Args:
        episodes: List of episode numbers
        values: List of metric values
        bin_size: Number of episodes per bin
        
    Returns:
        Tuple of (bin_centers, bin_means, bin_stds_lower, bin_stds_upper)
    """
    if bin_size <= 0 or len(episodes) == 0:
        return episodes, values, values, values
    
    binned_episodes = []
    binned_means = []
    binned_stds_lower = []
    binned_stds_upper = []
    
    for i in range(0, len(episodes), bin_size):
        bin_episodes = episodes[i:i+bin_size]
        bin_values = [v for v in values[i:i+bin_size] if not np.isnan(v)]
        
        if bin_values:
            bin_center = np.mean(bin_episodes)
            bin_mean = np.mean(bin_values)
            bin_std = np.std(bin_values)
            
            binned_episodes.append(bin_center)
            binned_means.append(bin_mean)
            binned_stds_lower.append(bin_mean - bin_std)
            binned_stds_upper.append(bin_mean + bin_std)
    
    return binned_episodes, binned_means, binned_stds_lower, binned_stds_upper


def generate_graph(
    episodes: List[int],
    values: List[float],
    metric_name: str,
    phase: str,
    training_type: str,
    output_dir: Path,
    file_format: str = "png",
    style: str = "seaborn",
    std_devs: Optional[List[float]] = None,
    statistics: Optional[Dict[str, float]] = None,
    smoothing_window: Optional[int] = None,
    bin_size: Optional[int] = None,
    show_raw: bool = False
) -> Path:
    """
    Generate a graph for a specific metric with research-standard visualization.
    
    Args:
        episodes: List of episode numbers
        values: List of metric values
        metric_name: Name of the metric
        phase: Training phase name
        training_type: Training type
        output_dir: Directory to save graph
        file_format: Output format (png, svg, pdf)
        style: Plot style
        std_devs: Optional standard deviations for error bars (ignored)
        statistics: Optional statistics to display
        smoothing_window: Optional window size for moving average (auto if None)
        bin_size: Optional bin size for aggregation (auto if None)
        show_raw: If True, show raw data points with low opacity
        
    Returns:
        Path to saved graph file
    """
    # Set style
    if style == "seaborn":
        sns.set_style("whitegrid")
    elif style == "ggplot":
        plt.style.use("ggplot")
    else:
        plt.style.use("default")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out NaN values for plotting
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    valid_episodes = [episodes[i] for i in valid_indices]
    valid_values = [values[i] for i in valid_indices]
    
    if not valid_values:
        print(f"Warning: No valid data for {metric_name} in {phase} for {training_type}")
        plt.close(fig)
        return None
    
    # Determine visualization strategy based on data size
    num_episodes = len(valid_episodes)
    
    # Auto-determine smoothing window if not specified
    if smoothing_window is None:
        if num_episodes > 1000:
            smoothing_window = max(50, num_episodes // 50)
        elif num_episodes > 100:
            smoothing_window = max(10, num_episodes // 20)
        else:
            smoothing_window = 5
    
    # Auto-determine bin size if not specified and data is very large
    if bin_size is None and num_episodes > 500:
        bin_size = max(10, num_episodes // 100)
    
    # Plot raw data if requested (with low opacity)
    if show_raw and num_episodes < 1000:
        ax.scatter(valid_episodes, valid_values, s=1, alpha=0.1, color='gray', label='Raw data')
    
    # Apply binning if specified and data is large
    if bin_size and num_episodes > 100:
        plot_episodes, plot_values, plot_lower, plot_upper = bin_episodes(
            valid_episodes, valid_values, bin_size
        )
        # Plot binned data with error bars
        ax.plot(plot_episodes, plot_values, linewidth=2.5, label=metric_name, color='#2E86AB', zorder=3)
        ax.fill_between(plot_episodes, plot_lower, plot_upper, alpha=0.2, color='#2E86AB', label='±1 std')
    else:
        # Apply moving average smoothing
        smoothed, lower_bounds, upper_bounds = compute_moving_average(valid_values, smoothing_window)
        
        # Plot smoothed line with confidence interval
        ax.plot(valid_episodes, smoothed, linewidth=2.5, label=metric_name, color='#2E86AB', zorder=3)
        ax.fill_between(valid_episodes, lower_bounds, upper_bounds, alpha=0.2, color='#2E86AB', label='±1 std')
    
    # Add trend line if statistics available
    if statistics and not np.isnan(statistics.get("trend", np.nan)):
        x_trend = np.array(valid_episodes)
        y_mean = statistics["mean"]
        trend = statistics["trend"]
        y_trend = y_mean + trend * (x_trend - np.mean(x_trend))
        ax.plot(valid_episodes, y_trend, '--', linewidth=2, alpha=0.7, color='#A23B72', label=f'Linear trend (slope={trend:.4f})', zorder=2)
    
    # Customize plot
    metric_display = metric_name.replace('_', ' ').title()
    phase_display = phase.replace('_', ' ').title()
    type_display = training_type.replace('_', ' ').title()
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(metric_display, fontsize=12)
    ax.set_title(f"{metric_display} - {phase_display} ({type_display})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text box if available
    if statistics:
        stats_text = f"Mean: {statistics['mean']:.4f}\n"
        stats_text += f"Std: {statistics['std']:.4f}\n"
        if not np.isnan(statistics.get('r_squared', np.nan)):
            stats_text += f"R²: {statistics['r_squared']:.4f}\n"
        if not np.isnan(statistics.get('trend', np.nan)):
            stats_text += f"Trend: {statistics['trend']:.4f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Create output directory structure
    phase_dir = output_dir / training_type / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    filename = f"{metric_name}_{phase}_{training_type}.{file_format}"
    file_path = phase_dir / filename
    fig.savefig(file_path, format=file_format, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated: {file_path}")
    return file_path


def generate_comparison_graph(
    all_metrics: Dict[str, Dict[int, List[Dict[str, Any]]]],
    metric_name: str,
    phase: str,
    output_dir: Path,
    file_format: str = "png",
    style: str = "seaborn"
) -> Optional[Path]:
    """
    Generate a comparison graph across multiple training types.
    
    Args:
        all_metrics: Dictionary mapping training_type to metrics_by_round
        metric_name: Name of metric to compare
        phase: Training phase name
        output_dir: Directory to save graph
        file_format: Output format
        style: Plot style
        
    Returns:
        Path to saved graph file or None
    """
    if style == "seaborn":
        sns.set_style("darkgrid")
    elif style == "ggplot":
        plt.style.use("ggplot")
    else:
        plt.style.use("default")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("husl", len(all_metrics))
    
    for idx, (training_type, metrics_by_round) in enumerate(all_metrics.items()):
        episodes, values, _ = extract_metric_series(metrics_by_round, metric_name, "mean")
        
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        if not valid_indices:
            continue
        
        valid_episodes = [episodes[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        
        # Apply smoothing for comparison graphs too
        num_episodes = len(valid_episodes)
        smoothing_window = max(10, num_episodes // 20) if num_episodes > 100 else 5
        
        smoothed, lower_bounds, upper_bounds = compute_moving_average(valid_values, smoothing_window)
        
        type_display = training_type.replace('_', ' ').title()
        ax.plot(valid_episodes, smoothed, linewidth=2.5, label=type_display, color=colors[idx], zorder=3)
        ax.fill_between(valid_episodes, lower_bounds, upper_bounds, alpha=0.15, color=colors[idx])
    
    metric_display = metric_name.replace('_', ' ').title()
    phase_display = phase.replace('_', ' ').title()
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(metric_display, fontsize=12)
    ax.set_title(f"{metric_display} Comparison - {phase_display}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save comparison graph
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metric_name}_comparison_{phase}.{file_format}"
    file_path = comparisons_dir / filename
    fig.savefig(file_path, format=file_format, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated comparison: {file_path}")
    return file_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate visualization graphs from training metrics files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/generate_metrics_graphs.py --type hand_only
  python scripts/generate_metrics_graphs.py --phase selfplay --metrics win_rate
  python scripts/generate_metrics_graphs.py --type hand_only --rounds 0,1,2,3
  python scripts/generate_metrics_graphs.py --format pdf --style seaborn

Valid training types: {', '.join(VALID_TRAINING_TYPES)}
Valid phases: {', '.join(VALID_PHASES)}
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=VALID_TRAINING_TYPES,
        help="Training type to analyze"
    )
    parser.add_argument(
        "--phase", "-p",
        help="Training phase to visualize (comma-separated for multiple)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./metrics_graphs"),
        help="Directory to save generated graphs (default: ./metrics_graphs)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--metrics", "-m",
        help="Comma-separated list of metrics to plot (default: all)"
    )
    parser.add_argument(
        "--rounds",
        help="Comma-separated list of rounds to include (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--style", "-s",
        choices=["default", "seaborn", "ggplot"],
        default="seaborn",
        help="Plot style (default: seaborn)"
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Don't combine trainee_first and trainee_second for validation phases"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing action_logs (default: current directory)"
    )
    parser.add_argument(
        "--comparisons",
        action="store_true",
        help="Generate comparison graphs across training types"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Window size for moving average smoothing (auto if not specified)"
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=None,
        help="Bin size for aggregating episodes (auto if not specified, only used for large datasets)"
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show raw data points with low opacity (only for datasets < 1000 episodes)"
    )
    
    args = parser.parse_args()
    
    # Determine training types to process
    if args.type:
        training_types = [args.type]
    else:
        training_types = VALID_TRAINING_TYPES
    
    # Determine phases to process
    if args.phase:
        phases = [p.strip() for p in args.phase.split(",")]
        for phase in phases:
            if phase not in VALID_PHASES:
                print(f"Error: Invalid phase '{phase}'. Valid phases: {', '.join(VALID_PHASES)}")
                return 1
    else:
        phases = VALID_PHASES
    
    # Determine rounds filter
    rounds_filter = None
    if args.rounds:
        try:
            rounds_filter = [int(r.strip()) for r in args.rounds.split(",")]
        except ValueError:
            print(f"Error: Invalid rounds format: {args.rounds}")
            return 1
    
    # Determine metrics to plot
    if args.metrics:
        requested_metrics = [m.strip() for m in args.metrics.split(",")]
        metrics_to_plot = []
        for metric in requested_metrics:
            if metric in METRIC_GROUPS:
                metrics_to_plot.extend(METRIC_GROUPS[metric])
            else:
                metrics_to_plot.append(metric)
    else:
        # Default: plot all common metrics
        metrics_to_plot = [
            "p1_win_rate", "p2_win_rate", "draw_rate",
            "loss", "p1_epsilon", "p2_epsilon",
            "p1_memory_size", "p2_memory_size",
            "p1_score", "p2_score", "episode_turns"
        ]
    
    # Process each training type and phase
    all_metrics_for_comparison = {}
    
    for training_type in training_types:
        print(f"\nProcessing {training_type}...")
        
        for phase in phases:
            print(f"  Phase: {phase}")
            
            # Aggregate metrics
            metrics_by_round = aggregate_phase_metrics(
                training_type,
                phase,
                args.base_dir,
                combine_positions=not args.no_combine,
                rounds_filter=rounds_filter
            )
            
            if not metrics_by_round:
                print(f"    No metrics found for {phase}")
                continue
            
            # Store for comparison graphs
            if args.comparisons:
                all_metrics_for_comparison[training_type] = all_metrics_for_comparison.get(training_type, {})
                all_metrics_for_comparison[training_type][phase] = metrics_by_round
            
            # Generate graphs for each metric
            for metric_name in metrics_to_plot:
                # Check if metric exists in any episode
                has_metric = any(
                    metric_name in ep
                    for episodes in metrics_by_round.values()
                    for ep in episodes
                )
                
                if not has_metric:
                    continue
                
                # Extract metric series (now returns episodes, not rounds)
                episodes, values, std_devs = extract_metric_series(
                    metrics_by_round, metric_name, "mean"
                )
                
                if not episodes or not any(not np.isnan(v) for v in values):
                    continue
                
                # Calculate statistics
                statistics = calculate_statistics(values)
                
                # Generate graph
                generate_graph(
                    episodes, values, metric_name, phase, training_type,
                    args.output_dir, args.format, args.style,
                    std_devs=std_devs, statistics=statistics,
                    smoothing_window=args.smoothing_window,
                    bin_size=args.bin_size,
                    show_raw=args.show_raw
                )
    
    # Generate comparison graphs if requested
    if args.comparisons and len(training_types) > 1:
        print("\nGenerating comparison graphs...")
        for phase in phases:
            phase_metrics = {}
            for training_type in training_types:
                if phase in all_metrics_for_comparison.get(training_type, {}):
                    phase_metrics[training_type] = all_metrics_for_comparison[training_type][phase]
            
            if len(phase_metrics) > 1:
                for metric_name in metrics_to_plot:
                    # Check if metric exists
                    has_metric = any(
                        metric_name in ep
                        for metrics_by_round in phase_metrics.values()
                        for episodes in metrics_by_round.values()
                        for ep in episodes
                    )
                    
                    if has_metric:
                        generate_comparison_graph(
                            phase_metrics, metric_name, phase,
                            args.output_dir, args.format, args.style
                        )
    
    print(f"\n✓ Graph generation complete. Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

