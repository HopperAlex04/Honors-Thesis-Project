#!/usr/bin/env python3
"""
Script to analyze agent improvement by tracking win rates against fixed opponents.

This script analyzes validation metrics to show how agents improve over training rounds
when playing against fixed opponents (heuristic, randomized, previous models, etc.).

Usage:
    python scripts/analyze_improvement.py
    python scripts/analyze_improvement.py --type hand_only
    python scripts/analyze_improvement.py --opponent heuristic --type all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Valid training types
VALID_TRAINING_TYPES = [
    "hand_only",
    "opponent_field_only",
    "no_features",
    "both_features",  # Legacy name, kept for backward compatibility
    "all_features",
    "scores"
]

# Valid opponent types
VALID_OPPONENTS = [
    "vs_previous",
    "vs_randomized",
    "vs_heuristic",
    "vs_gapmaximizer"
]


def extract_round_number(filename: str) -> Optional[int]:
    """Extract round number from metrics filename."""
    import re
    match = re.search(r'_round_(\d+)_', filename)
    return int(match.group(1)) if match else None


def load_validation_metrics(
    base_dir: Path,
    training_type: str,
    opponent_type: str
) -> Dict[int, Dict[str, float]]:
    """
    Load validation metrics for a specific training type and opponent.
    
    Args:
        base_dir: Base directory containing action_logs
        training_type: Training type (e.g., "hand_only")
        opponent_type: Opponent type (e.g., "vs_heuristic")
        
    Returns:
        Dictionary mapping round number to metrics (win_rate, etc.)
    """
    log_dir = base_dir / "action_logs" / training_type
    
    if not log_dir.exists():
        return {}
    
    metrics_by_round = {}
    
    # Pattern for validation files (combines trainee_first and trainee_second)
    first_pattern = f"metrics_{training_type}_round_*_{opponent_type}_trainee_first.jsonl"
    second_pattern = f"metrics_{training_type}_round_*_{opponent_type}_trainee_second.jsonl"
    
    # Find all matching files
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
    
    # Process each round
    all_rounds = set(first_files.keys()) | set(second_files.keys())
    
    for round_num in sorted(all_rounds):
        # Track last episode data from each file (cumulative counts are in last episode)
        first_last_episode = None
        second_last_episode = None
        first_episodes = 0
        second_episodes = 0
        
        # Load first file (trainee as p1) - get last episode
        if round_num in first_files:
            with open(first_files[round_num], 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        first_last_episode = data  # Keep updating to get last episode
                        first_episodes += 1
                    except json.JSONDecodeError:
                        continue
        
        # Load second file (trainee as p2) - get last episode
        if round_num in second_files:
            with open(second_files[round_num], 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        second_last_episode = data  # Keep updating to get last episode
                        second_episodes += 1
                    except json.JSONDecodeError:
                        continue
        
        # Calculate metrics from last episodes (which contain cumulative counts)
        total_episodes = first_episodes + second_episodes
        
        if total_episodes > 0:
            # From first file: trainee is p1
            trainee_wins_first = first_last_episode.get('p1_wins', 0) if first_last_episode else 0
            opponent_wins_first = first_last_episode.get('p2_wins', 0) if first_last_episode else 0
            draws_first = first_last_episode.get('draws', 0) if first_last_episode else 0
            
            # From second file: trainee is p2 (so swap p1/p2)
            trainee_wins_second = second_last_episode.get('p2_wins', 0) if second_last_episode else 0
            opponent_wins_second = second_last_episode.get('p1_wins', 0) if second_last_episode else 0
            draws_second = second_last_episode.get('draws', 0) if second_last_episode else 0
            
            # Total wins across both files
            total_trainee_wins = trainee_wins_first + trainee_wins_second
            total_opponent_wins = opponent_wins_first + opponent_wins_second
            total_draws = draws_first + draws_second
            total_games = total_trainee_wins + total_opponent_wins + total_draws
            
            if total_games > 0:
                trainee_win_rate = total_trainee_wins / total_games
                opponent_win_rate = total_opponent_wins / total_games
                draw_rate = total_draws / total_games
            else:
                trainee_win_rate = 0.0
                opponent_win_rate = 0.0
                draw_rate = 0.0
            
            metrics_by_round[round_num] = {
                'trainee_win_rate': trainee_win_rate,
                'opponent_win_rate': opponent_win_rate,
                'draw_rate': draw_rate,
                'total_episodes': total_episodes,
                'total_games': total_games,
                'trainee_wins': total_trainee_wins,
                'opponent_wins': total_opponent_wins,
                'draws': total_draws
            }
    
    return metrics_by_round


def calculate_improvement_trend(win_rates: List[float]) -> Dict[str, float]:
    """
    Calculate improvement trend statistics.
    
    Args:
        win_rates: List of win rates over rounds
        
    Returns:
        Dictionary with trend statistics
    """
    if len(win_rates) < 2:
        return {
            'trend': 0.0,
            'r_squared': 0.0,
            'improvement': 0.0,
            'final_rate': win_rates[0] if win_rates else 0.0
        }
    
    x = np.arange(len(win_rates))
    valid_indices = [i for i, v in enumerate(win_rates) if not np.isnan(v)]
    
    if len(valid_indices) < 2:
        return {
            'trend': 0.0,
            'r_squared': 0.0,
            'improvement': 0.0,
            'final_rate': win_rates[-1] if win_rates else 0.0
        }
    
    valid_x = np.array([x[i] for i in valid_indices])
    valid_y = np.array([win_rates[i] for i in valid_indices])
    
    # Linear regression
    coeffs = np.polyfit(valid_x, valid_y, 1)
    trend = coeffs[0]  # Slope
    
    # R-squared
    y_pred = np.polyval(coeffs, valid_x)
    ss_res = np.sum((valid_y - y_pred) ** 2)
    ss_tot = np.sum((valid_y - np.mean(valid_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Overall improvement (final - initial)
    improvement = valid_y[-1] - valid_y[0] if len(valid_y) > 0 else 0.0
    
    return {
        'trend': float(trend),
        'r_squared': float(r_squared),
        'improvement': float(improvement),
        'final_rate': float(valid_y[-1]),
        'initial_rate': float(valid_y[0])
    }


def plot_improvement_analysis(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    opponent_type: str,
    output_dir: Path,
    file_format: str = "png"
) -> Path:
    """
    Plot improvement analysis across training types.
    
    Args:
        all_metrics: Dictionary mapping training_type to metrics_by_round
        opponent_type: Opponent type being analyzed
        output_dir: Directory to save plot
        file_format: Output format
        
    Returns:
        Path to saved plot
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Agent Improvement Analysis - {opponent_type.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(all_metrics))
    
    # Plot 1: Win rate over rounds
    ax1 = axes[0, 0]
    for idx, (training_type, metrics_by_round) in enumerate(all_metrics.items()):
        rounds = sorted(metrics_by_round.keys())
        win_rates = [metrics_by_round[r]['trainee_win_rate'] for r in rounds]
        
        type_display = training_type.replace('_', ' ').title()
        ax1.plot(rounds, win_rates, marker='o', linewidth=2.5, markersize=8,
                label=type_display, color=colors[idx], zorder=3)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (Random)')
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Trainee Win Rate', fontsize=12)
    ax1.set_title('Win Rate Over Training Rounds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Plot 2: Improvement trend
    ax2 = axes[0, 1]
    trends = []
    labels = []
    for training_type, metrics_by_round in all_metrics.items():
        rounds = sorted(metrics_by_round.keys())
        win_rates = [metrics_by_round[r]['trainee_win_rate'] for r in rounds]
        stats = calculate_improvement_trend(win_rates)
        
        trends.append(stats['trend'])
        labels.append(training_type.replace('_', ' ').title())
    
    bars = ax2.barh(labels, trends, color=colors[:len(trends)])
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Improvement Trend (Slope)', fontsize=12)
    ax2.set_title('Learning Rate (Positive = Improving)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, trend) in enumerate(zip(bars, trends)):
        width = bar.get_width()
        ax2.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                f'{trend:+.4f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10)
    
    # Plot 3: Overall improvement (final - initial)
    ax3 = axes[1, 0]
    improvements = []
    for training_type, metrics_by_round in all_metrics.items():
        rounds = sorted(metrics_by_round.keys())
        win_rates = [metrics_by_round[r]['trainee_win_rate'] for r in rounds]
        stats = calculate_improvement_trend(win_rates)
        improvements.append(stats['improvement'])
    
    bars = ax3.barh(labels, improvements, color=colors[:len(improvements)])
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Overall Improvement (Final - Initial)', fontsize=12)
    ax3.set_title('Total Improvement Over Training', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax3.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{imp:+.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10)
    
    # Plot 4: Final performance
    ax4 = axes[1, 1]
    final_rates = []
    for training_type, metrics_by_round in all_metrics.items():
        rounds = sorted(metrics_by_round.keys())
        if rounds:
            final_rates.append(metrics_by_round[rounds[-1]]['trainee_win_rate'])
        else:
            final_rates.append(0.0)
    
    bars = ax4.barh(labels, final_rates, color=colors[:len(final_rates)])
    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (Random)')
    ax4.set_xlabel('Final Win Rate', fontsize=12)
    ax4.set_title('Final Performance', fontsize=13, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend()
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, final_rates)):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"improvement_analysis_{opponent_type}.{file_format}"
    file_path = output_dir / filename
    fig.savefig(file_path, format=file_format, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated: {file_path}")
    return file_path


def analyze_regression_frequency(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    opponent_type: str
) -> None:
    """
    Analyze how frequently each generation loses to the previous iteration.
    
    This is specifically for vs_previous validation, showing regression patterns.
    """
    if opponent_type != "vs_previous":
        return  # Only analyze regression for vs_previous
    
    print(f"\n{'='*80}")
    print(f"REGRESSION FREQUENCY ANALYSIS: {opponent_type.replace('_', ' ').upper()}")
    print(f"{'='*80}\n")
    
    for training_type, metrics_by_round in sorted(all_metrics.items()):
        rounds = sorted(metrics_by_round.keys())
        if len(rounds) < 2:
            continue  # Need at least 2 rounds to compare
        
        type_display = training_type.replace('_', ' ').title()
        print(f"{type_display}:")
        print(f"  Total Rounds: {len(rounds)}")
        print(f"  Rounds Compared: {len(rounds) - 1} (round 0 has no previous to compare)")
        print()
        
        # Track regression statistics
        regressions = []  # List of (round, win_rate) tuples where regression occurred
        improvements = []  # List of (round, win_rate) tuples where improvement occurred
        marginal = []  # List of (round, win_rate) tuples where win rate is 0.50-0.55 (barely winning)
        
        # Compare each round to previous (starting from round 1)
        for i in range(1, len(rounds)):
            current_round = rounds[i]
            win_rate = metrics_by_round[current_round]['trainee_win_rate']
            trainee_wins = metrics_by_round[current_round]['trainee_wins']
            opponent_wins = metrics_by_round[current_round]['opponent_wins']
            total_games = metrics_by_round[current_round]['total_games']
            
            if win_rate < 0.50:
                # Regression: losing to previous model
                regressions.append((current_round, win_rate, trainee_wins, opponent_wins, total_games))
            elif win_rate < 0.55:
                # Marginal: barely beating previous model
                marginal.append((current_round, win_rate, trainee_wins, opponent_wins, total_games))
            else:
                # Improvement: clearly beating previous model
                improvements.append((current_round, win_rate, trainee_wins, opponent_wins, total_games))
        
        # Print regression details
        if regressions:
            print(f"  ⚠ REGRESSIONS DETECTED: {len(regressions)}/{len(rounds)-1} rounds ({len(regressions)/(len(rounds)-1)*100:.1f}%)")
            print(f"     Rounds that lost to previous model:")
            for round_num, win_rate, trainee_wins, opponent_wins, total_games in regressions:
                print(f"       Round {round_num}: {win_rate:.1%} win rate ({trainee_wins}W / {opponent_wins}L / {total_games} games)")
            print()
        else:
            print(f"  ✓ No regressions detected (all rounds beat previous model)")
            print()
        
        # Print marginal cases
        if marginal:
            print(f"  ⚠ MARGINAL WINS: {len(marginal)}/{len(rounds)-1} rounds ({len(marginal)/(len(rounds)-1)*100:.1f}%)")
            print(f"     Rounds barely beating previous model (50-55% win rate):")
            for round_num, win_rate, trainee_wins, opponent_wins, total_games in marginal:
                print(f"       Round {round_num}: {win_rate:.1%} win rate ({trainee_wins}W / {opponent_wins}L / {total_games} games)")
            print()
        
        # Print improvement summary
        if improvements:
            improvement_rate = len(improvements) / (len(rounds) - 1) * 100
            print(f"  ✓ CLEAR IMPROVEMENTS: {len(improvements)}/{len(rounds)-1} rounds ({improvement_rate:.1f}%)")
            print(f"     Rounds clearly beating previous model (>55% win rate)")
            if len(improvements) <= 5:
                # Show details if few improvements
                for round_num, win_rate, trainee_wins, opponent_wins, total_games in improvements:
                    print(f"       Round {round_num}: {win_rate:.1%} win rate ({trainee_wins}W / {opponent_wins}L / {total_games} games)")
            else:
                # Show summary if many improvements
                avg_win_rate = sum(wr for _, wr, _, _, _ in improvements) / len(improvements)
                print(f"       Average win rate: {avg_win_rate:.1%}")
            print()
        
        # Overall statistics
        total_comparisons = len(rounds) - 1
        regression_rate = len(regressions) / total_comparisons if total_comparisons > 0 else 0
        marginal_rate = len(marginal) / total_comparisons if total_comparisons > 0 else 0
        improvement_rate = len(improvements) / total_comparisons if total_comparisons > 0 else 0
        
        print(f"  Summary Statistics:")
        print(f"    Regression Rate:    {regression_rate:.1%} ({len(regressions)}/{total_comparisons})")
        print(f"    Marginal Win Rate:  {marginal_rate:.1%} ({len(marginal)}/{total_comparisons})")
        print(f"    Clear Win Rate:     {improvement_rate:.1%} ({len(improvements)}/{total_comparisons})")
        print()


def print_improvement_summary(
    all_metrics: Dict[str, Dict[int, Dict[str, float]]],
    opponent_type: str
) -> None:
    """Print a summary of improvement statistics."""
    print(f"\n{'='*80}")
    print(f"IMPROVEMENT ANALYSIS: {opponent_type.replace('_', ' ').upper()}")
    print(f"{'='*80}\n")
    
    for training_type, metrics_by_round in sorted(all_metrics.items()):
        rounds = sorted(metrics_by_round.keys())
        if not rounds:
            continue
        
        win_rates = [metrics_by_round[r]['trainee_win_rate'] for r in rounds]
        stats = calculate_improvement_trend(win_rates)
        
        type_display = training_type.replace('_', ' ').title()
        print(f"{type_display}:")
        print(f"  Initial Win Rate: {stats['initial_rate']:.3f}")
        print(f"  Final Win Rate:   {stats['final_rate']:.3f}")
        print(f"  Overall Improvement: {stats['improvement']:+.3f} ({stats['improvement']*100:+.1f}%)")
        print(f"  Learning Trend: {stats['trend']:+.6f} per round (R² = {stats['r_squared']:.3f})")
        
        if stats['improvement'] > 0.05:
            print(f"  ✓ Significant improvement detected!")
        elif stats['improvement'] > 0:
            print(f"  → Modest improvement")
        elif stats['improvement'] > -0.05:
            print(f"  → Stable performance")
        else:
            print(f"  ⚠ Performance degradation detected!")
        
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze agent improvement by tracking win rates against fixed opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/analyze_improvement.py
  python scripts/analyze_improvement.py --type hand_only
  python scripts/analyze_improvement.py --opponent vs_heuristic
  python scripts/analyze_improvement.py --type all --opponent vs_gapmaximizer

Valid training types: {', '.join(VALID_TRAINING_TYPES)}
Valid opponents: {', '.join(VALID_OPPONENTS)}
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=VALID_TRAINING_TYPES + ["all"],
        default="all",
        help="Training type to analyze (default: all)"
    )
    parser.add_argument(
        "--opponent", "-o",
        choices=VALID_OPPONENTS + ["all"],
        default="all",
        help="Opponent type to analyze (default: all)"
    )
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        default=Path("."),
        help="Base directory containing action_logs (default: current directory)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=Path,
        default=Path("improvement_analysis"),
        help="Directory to save analysis plots (default: improvement_analysis)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format for plots (default: png)"
    )
    
    args = parser.parse_args()
    
    # Determine training types to analyze
    if args.type == "all":
        training_types = VALID_TRAINING_TYPES
    else:
        training_types = [args.type]
    
    # Determine opponents to analyze
    if args.opponent == "all":
        opponents = VALID_OPPONENTS
    else:
        opponents = [args.opponent]
    
    # Analyze each opponent type
    for opponent_type in opponents:
        all_metrics = {}
        
        for training_type in training_types:
            metrics = load_validation_metrics(args.base_dir, training_type, opponent_type)
            if metrics:
                all_metrics[training_type] = metrics
        
        if not all_metrics:
            print(f"No data found for {opponent_type}")
            continue
        
        # Print regression frequency analysis (only for vs_previous)
        if opponent_type == "vs_previous":
            analyze_regression_frequency(all_metrics, opponent_type)
        
        # Print summary
        print_improvement_summary(all_metrics, opponent_type)
        
        # Generate plots
        plot_improvement_analysis(all_metrics, opponent_type, args.output_dir, args.format)
    
    print(f"\n✓ Analysis complete. Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

