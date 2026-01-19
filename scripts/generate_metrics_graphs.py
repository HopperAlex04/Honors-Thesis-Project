#!/usr/bin/env python3
"""
Generate visualization graphs from experiment metrics.

This script generates graphs from training metrics stored in the experiment
manager's directory structure. It supports both round-level and episode-level
visualizations, and can compare multiple experiments.

Usage:
    python scripts/generate_metrics_graphs.py                      # Current experiment
    python scripts/generate_metrics_graphs.py --experiment NAME    # Specific experiment
    python scripts/generate_metrics_graphs.py --compare EXP1 EXP2  # Compare experiments
    python scripts/generate_metrics_graphs.py --episodic           # Episode-level graphs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CURRENT_EXPERIMENT_FILE = EXPERIMENTS_DIR / ".current_experiment"

# Valid phases for validation
VALIDATION_PHASES = ["vs_randomized", "vs_gapmaximizer"]
SELFPLAY_PHASE = "selfplay"


def get_current_experiment() -> Optional[Path]:
    """Get the path to the current experiment."""
    if not CURRENT_EXPERIMENT_FILE.exists():
        return None
    
    with open(CURRENT_EXPERIMENT_FILE, 'r') as f:
        exp_path = Path(f.read().strip())
    
    if exp_path.exists():
        return exp_path
    return None


def find_experiment(name: str) -> Optional[Path]:
    """Find an experiment by name (partial match)."""
    if not EXPERIMENTS_DIR.exists():
        return None
    
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and name in exp_dir.name:
            return exp_dir
    
    return None


def get_metrics_dir(run_path: Path) -> Optional[Path]:
    """Get the metrics directory for a run (metrics_logs or action_logs fallback)."""
    metrics_logs = run_path / "metrics_logs"
    if metrics_logs.exists():
        return metrics_logs
    
    action_logs = run_path / "action_logs"
    if action_logs.exists():
        return action_logs
    
    return None


def parse_metrics_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL metrics file and return list of episode metrics."""
    metrics = []
    if not file_path.exists():
        return metrics
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return metrics


def extract_round_number(filename: str) -> Optional[int]:
    """Extract round number from filename like 'metrics_round_3_selfplay.jsonl'."""
    import re
    match = re.search(r'round_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def load_run_metrics(run_path: Path) -> Dict[str, Any]:
    """
    Load all metrics from a single run.
    
    Returns dict with:
        - selfplay: {round: [episode_metrics]}
        - vs_randomized: {round: [episode_metrics]}  
        - vs_gapmaximizer: {round: [episode_metrics]}
    """
    metrics_dir = get_metrics_dir(run_path)
    if not metrics_dir:
        return {}
    
    result = {
        "selfplay": {},
        "vs_randomized": {},
        "vs_gapmaximizer": {},
    }
    
    for metrics_file in metrics_dir.glob("metrics_*.jsonl"):
        filename = metrics_file.name
        round_num = extract_round_number(filename)
        
        if round_num is None:
            continue
        
        episodes = parse_metrics_file(metrics_file)
        if not episodes:
            continue
        
        if "selfplay" in filename:
            result["selfplay"][round_num] = episodes
        elif "vs_randomized" in filename:
            # Combine trainee_first and trainee_second
            if round_num not in result["vs_randomized"]:
                result["vs_randomized"][round_num] = []
            result["vs_randomized"][round_num].extend(episodes)
        elif "vs_gapmaximizer" in filename:
            if round_num not in result["vs_gapmaximizer"]:
                result["vs_gapmaximizer"][round_num] = []
            result["vs_gapmaximizer"][round_num].extend(episodes)
    
    return result


def load_experiment_metrics(experiment_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load metrics from all completed runs in an experiment.
    
    Returns dict mapping run_id -> run_metrics
    """
    runs_dir = experiment_path / "runs"
    if not runs_dir.exists():
        return {}
    
    result = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Check if run has metrics
        metrics = load_run_metrics(run_dir)
        if metrics:
            result[run_dir.name] = metrics
    
    return result


def calculate_round_win_rates(
    experiment_metrics: Dict[str, Dict[str, Any]],
    phase: str = "vs_randomized"
) -> Dict[int, List[float]]:
    """
    Calculate win rates per round across all runs.
    
    Returns dict mapping round_number -> list of win rates (one per run)
    """
    round_win_rates: Dict[int, List[float]] = {}
    
    for run_id, run_metrics in experiment_metrics.items():
        if phase not in run_metrics:
            continue
        
        for round_num, episodes in run_metrics[phase].items():
            if not episodes:
                continue
            
            # Get final episode which has cumulative stats
            last_ep = episodes[-1]
            
            # Determine trainee win rate based on position
            if "trainee" in str(last_ep.get("p1_name", "")):
                win_rate = last_ep.get("p1_win_rate", 0)
            elif "trainee" in str(last_ep.get("p2_name", "")):
                win_rate = last_ep.get("p2_win_rate", 0)
            else:
                # Default: use p1_wins and p2_wins to calculate
                p1_wins = last_ep.get("p1_wins", 0)
                p2_wins = last_ep.get("p2_wins", 0)
                total = p1_wins + p2_wins
                # Assume PlayerAgent is the trainee
                if "PlayerAgent" in str(last_ep.get("p1_name", "")):
                    win_rate = p1_wins / total if total > 0 else 0
                else:
                    win_rate = p2_wins / total if total > 0 else 0
            
            if round_num not in round_win_rates:
                round_win_rates[round_num] = []
            round_win_rates[round_num].append(win_rate)
    
    return round_win_rates


def calculate_episodic_metrics(
    experiment_metrics: Dict[str, Dict[str, Any]],
    phase: str = "selfplay",
    metric: str = "loss"
) -> Tuple[List[int], List[float], List[float]]:
    """
    Calculate episode-level metrics with mean and std across runs.
    
    Returns:
        - episode_numbers: List of global episode numbers
        - means: Mean value at each episode
        - stds: Std value at each episode
    """
    # Collect all data by global episode number
    episode_data: Dict[int, List[float]] = {}
    
    for run_id, run_metrics in experiment_metrics.items():
        if phase not in run_metrics:
            continue
        
        global_episode = 0
        for round_num in sorted(run_metrics[phase].keys()):
            episodes = run_metrics[phase][round_num]
            
            for ep in episodes:
                value = ep.get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    if global_episode not in episode_data:
                        episode_data[global_episode] = []
                    episode_data[global_episode].append(value)
                global_episode += 1
    
    if not episode_data:
        return [], [], []
    
    episodes = sorted(episode_data.keys())
    means = [np.mean(episode_data[e]) for e in episodes]
    stds = [np.std(episode_data[e]) if len(episode_data[e]) > 1 else 0 for e in episodes]
    
    return episodes, means, stds


def generate_round_win_rate_graph(
    experiment_path: Path,
    experiment_metrics: Dict[str, Dict[str, Any]],
    output_dir: Path,
    phases: List[str] = None
) -> List[Path]:
    """Generate win rate by round graphs."""
    if phases is None:
        phases = VALIDATION_PHASES
    
    generated = []
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, len(phases), figsize=(7 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, phase in enumerate(phases):
        ax = axes[idx]
        round_win_rates = calculate_round_win_rates(experiment_metrics, phase)
        
        if not round_win_rates:
            ax.text(0.5, 0.5, f"No data for {phase}", ha='center', va='center', transform=ax.transAxes)
            continue
        
        rounds = sorted(round_win_rates.keys())
        means = [np.mean(round_win_rates[r]) for r in rounds]
        stds = [np.std(round_win_rates[r]) if len(round_win_rates[r]) > 1 else 0 for r in rounds]
        
        # Plot mean line with confidence band
        ax.plot(rounds, means, marker='o', linewidth=2, markersize=8, color=colors[0], label='Mean')
        ax.fill_between(rounds, 
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       alpha=0.2, color=colors[0])
        
        # Plot individual runs
        for run_id, run_metrics in experiment_metrics.items():
            if phase not in run_metrics:
                continue
            
            run_rounds = []
            run_rates = []
            for round_num in sorted(run_metrics[phase].keys()):
                episodes = run_metrics[phase][round_num]
                if episodes:
                    last_ep = episodes[-1]
                    p1_wins = last_ep.get("p1_wins", 0)
                    p2_wins = last_ep.get("p2_wins", 0)
                    total = p1_wins + p2_wins
                    if "PlayerAgent" in str(last_ep.get("p1_name", "")):
                        wr = p1_wins / total if total > 0 else 0
                    else:
                        wr = p2_wins / total if total > 0 else 0
                    run_rounds.append(round_num)
                    run_rates.append(wr)
            
            if run_rounds:
                ax.plot(run_rounds, run_rates, alpha=0.3, linewidth=1, color=colors[0])
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title(f'Win Rate {phase.replace("_", " ").title()}', fontsize=14)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Win Rate by Round - {experiment_path.name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "round_win_rates.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(output_file)
    
    return generated


def generate_episodic_loss_graph(
    experiment_path: Path,
    experiment_metrics: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> List[Path]:
    """Generate episode-level loss graph."""
    generated = []
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    episodes, means, stds = calculate_episodic_metrics(experiment_metrics, "selfplay", "loss")
    
    if not episodes:
        print("No loss data found for episodic graph")
        return generated
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Apply smoothing for readability
    window = min(50, len(means) // 10) if len(means) > 100 else 1
    if window > 1:
        smoothed_means = np.convolve(means, np.ones(window)/window, mode='valid')
        smoothed_episodes = episodes[window-1:]
    else:
        smoothed_means = means
        smoothed_episodes = episodes
    
    ax.plot(smoothed_episodes, smoothed_means, linewidth=1.5, color='#3498db', label='Loss (smoothed)')
    
    # Show raw data with low opacity if not too many points
    if len(episodes) < 2000:
        ax.plot(episodes, means, alpha=0.2, linewidth=0.5, color='#3498db')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss Over Episodes - {experiment_path.name}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "episodic_loss.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(output_file)
    
    return generated


def generate_episodic_epsilon_graph(
    experiment_path: Path,
    experiment_metrics: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> List[Path]:
    """Generate episode-level epsilon decay graph."""
    generated = []
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    episodes, means, stds = calculate_episodic_metrics(experiment_metrics, "selfplay", "p1_epsilon")
    
    if not episodes:
        print("No epsilon data found for episodic graph")
        return generated
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(episodes, means, linewidth=1.5, color='#e74c3c', label='Epsilon')
    ax.fill_between(episodes,
                   [m - s for m, s in zip(means, stds)],
                   [m + s for m, s in zip(means, stds)],
                   alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_title(f'Exploration Rate (Epsilon) Over Episodes - {experiment_path.name}', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "episodic_epsilon.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(output_file)
    
    return generated


def generate_comparison_graph(
    experiments: List[Tuple[str, Path]],
    output_dir: Path,
    phase: str = "vs_randomized"
) -> List[Path]:
    """Generate comparison graph across multiple experiments."""
    generated = []
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (exp_name, exp_path) in enumerate(experiments):
        exp_metrics = load_experiment_metrics(exp_path)
        if not exp_metrics:
            continue
        
        round_win_rates = calculate_round_win_rates(exp_metrics, phase)
        if not round_win_rates:
            continue
        
        rounds = sorted(round_win_rates.keys())
        means = [np.mean(round_win_rates[r]) for r in rounds]
        
        color = colors[idx % len(colors)]
        ax.plot(rounds, means, marker='o', linewidth=2, markersize=8, 
                color=color, label=exp_name)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title(f'Experiment Comparison - {phase.replace("_", " ").title()}', fontsize=14)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"comparison_{phase}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(output_file)
    
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization graphs from experiment metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_metrics_graphs.py                      # Current experiment
    python scripts/generate_metrics_graphs.py --experiment sparse  # Match by name
    python scripts/generate_metrics_graphs.py --compare exp1 exp2  # Compare experiments
    python scripts/generate_metrics_graphs.py --episodic           # Episode-level graphs
    python scripts/generate_metrics_graphs.py --all                # All graph types
"""
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Experiment name (partial match supported)"
    )
    parser.add_argument(
        "--compare", "-c",
        nargs="+",
        type=str,
        help="Compare multiple experiments by name"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for graphs (default: experiment/analysis/graphs)"
    )
    parser.add_argument(
        "--episodic",
        action="store_true",
        help="Generate episode-level graphs (loss, epsilon)"
    )
    parser.add_argument(
        "--rounds",
        action="store_true",
        help="Generate round-level win rate graphs"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all graph types"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["vs_randomized", "vs_gapmaximizer", "both"],
        default="both",
        help="Validation phase to graph (default: both)"
    )
    
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        experiments = []
        for name in args.compare:
            exp_path = find_experiment(name)
            if exp_path:
                # Extract display name from metadata or use folder name
                metadata_file = exp_path / "experiment_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    display_name = metadata.get("display_name", exp_path.name)
                else:
                    display_name = exp_path.name
                experiments.append((display_name, exp_path))
            else:
                print(f"Warning: Experiment '{name}' not found")
        
        if len(experiments) < 2:
            print("Error: Need at least 2 experiments to compare")
            return 1
        
        # Output to first experiment's analysis folder
        output_dir = Path(args.output_dir) if args.output_dir else experiments[0][1] / "analysis" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        phases = VALIDATION_PHASES if args.phase == "both" else [args.phase]
        
        for phase in phases:
            generated = generate_comparison_graph(experiments, output_dir, phase)
            for g in generated:
                print(f"Generated: {g}")
        
        return 0
    
    # Single experiment mode
    if args.experiment:
        experiment_path = find_experiment(args.experiment)
        if not experiment_path:
            print(f"Error: Experiment '{args.experiment}' not found")
            return 1
    else:
        experiment_path = get_current_experiment()
        if not experiment_path:
            print("Error: No current experiment. Use --experiment to specify one.")
            return 1
    
    print(f"Generating graphs for: {experiment_path.name}")
    
    # Load metrics
    experiment_metrics = load_experiment_metrics(experiment_path)
    if not experiment_metrics:
        print("Error: No metrics found in experiment")
        return 1
    
    print(f"Found {len(experiment_metrics)} runs with metrics")
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else experiment_path / "analysis" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which graphs to generate
    generate_rounds = args.rounds or args.all or (not args.episodic)
    generate_episodic = args.episodic or args.all
    
    phases = VALIDATION_PHASES if args.phase == "both" else [args.phase]
    
    generated_files = []
    
    # Generate round-level graphs
    if generate_rounds:
        generated = generate_round_win_rate_graph(experiment_path, experiment_metrics, output_dir, phases)
        generated_files.extend(generated)
    
    # Generate episodic graphs
    if generate_episodic:
        generated = generate_episodic_loss_graph(experiment_path, experiment_metrics, output_dir)
        generated_files.extend(generated)
        
        generated = generate_episodic_epsilon_graph(experiment_path, experiment_metrics, output_dir)
        generated_files.extend(generated)
    
    # Print summary
    print(f"\nGenerated {len(generated_files)} graphs:")
    for f in generated_files:
        print(f"  {f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
