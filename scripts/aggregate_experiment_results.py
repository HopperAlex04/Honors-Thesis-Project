#!/usr/bin/env python3
"""
Aggregate Experiment Results - Combine and analyze results across all runs.

This script aggregates metrics from all completed runs in an experiment,
calculates statistics, and generates comparison visualizations suitable
for thesis-quality figures.

Features:
- Aggregates win rates across runs for each network type
- Calculates mean, std, 95% CI for each metric
- Performs statistical tests (t-tests, ANOVA)
- Generates publication-quality comparison graphs
- Exports results in various formats

Usage:
    python scripts/aggregate_experiment_results.py
    python scripts/aggregate_experiment_results.py --graphs
    python scripts/aggregate_experiment_results.py --export csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiment_manager import ExperimentManager, NETWORK_TYPES


def load_run_metrics(run_path: Path, network_type: str) -> Dict[str, List[float]]:
    """
    Load all metrics from a completed run.
    
    Args:
        run_path: Path to run directory
        network_type: Network type for this run
        
    Returns:
        Dictionary mapping metric names to lists of values across rounds
    """
    metrics = {
        "win_rates_by_round": [],
        "losses_by_episode": [],
        "final_win_rate": None,
    }
    
    action_logs = run_path / "action_logs"
    if not action_logs.exists():
        return metrics
    
    # The new training stores logs under round_X directories or directly
    # Look for validation metrics files
    for subdir in action_logs.iterdir():
        if not subdir.is_dir():
            continue
        
        # Find validation files (vs_randomized or vs_gapmaximizer)
        validation_files = sorted(subdir.glob("metrics_round_*_vs_*.jsonl"))
        
        for vf in validation_files:
            # Read the last line to get final round stats
            with open(vf, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_metrics = json.loads(lines[-1])
                    win_rate = last_metrics.get("p1_win_rate", 0)
                    metrics["win_rates_by_round"].append(win_rate)
        
        # Find selfplay files for loss
        selfplay_files = sorted(subdir.glob("metrics_round_*_selfplay.jsonl"))
        
        for sf in selfplay_files:
            with open(sf, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("loss") is not None:
                        metrics["losses_by_episode"].append(data["loss"])
    
    # Set final win rate
    if metrics["win_rates_by_round"]:
        metrics["final_win_rate"] = metrics["win_rates_by_round"][-1]
    
    return metrics


def aggregate_metrics_by_type(
    manager: ExperimentManager,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate metrics across all runs for each network type.
    
    Args:
        manager: ExperimentManager instance
        
    Returns:
        Dictionary mapping network type to aggregated statistics
    """
    aggregated = {}
    
    for network_type in NETWORK_TYPES:
        runs = manager.get_runs_by_type(network_type)
        completed_runs = [r for r in runs if r["status"] == "completed"]
        
        if not completed_runs:
            aggregated[network_type] = {
                "n_runs": 0,
                "error": "No completed runs"
            }
            continue
        
        # Collect final win rates from run metadata
        final_win_rates = []
        all_round_win_rates = []  # List of lists (one per run)
        
        for run_info in completed_runs:
            run_path = manager.get_run_path(run_info["run_id"])
            run_metrics = load_run_metrics(run_path, network_type)
            
            # Prefer metadata if available
            if run_info.get("final_metrics", {}).get("final_win_rate") is not None:
                final_win_rates.append(run_info["final_metrics"]["final_win_rate"])
            elif run_metrics["final_win_rate"] is not None:
                final_win_rates.append(run_metrics["final_win_rate"])
            
            if run_metrics["win_rates_by_round"]:
                all_round_win_rates.append(run_metrics["win_rates_by_round"])
        
        if not final_win_rates:
            aggregated[network_type] = {
                "n_runs": len(completed_runs),
                "error": "No win rate data found"
            }
            continue
        
        # Calculate statistics
        n = len(final_win_rates)
        mean = np.mean(final_win_rates)
        std = np.std(final_win_rates, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        
        # 95% CI using t-distribution
        if n > 1:
            from scipy import stats
            t_crit = stats.t.ppf(0.975, n - 1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean
        
        aggregated[network_type] = {
            "n_runs": n,
            "final_win_rates": final_win_rates,
            "mean": float(mean),
            "std": float(std),
            "se": float(se),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "min": float(np.min(final_win_rates)),
            "max": float(np.max(final_win_rates)),
            "median": float(np.median(final_win_rates)),
            "round_win_rates": all_round_win_rates,
        }
    
    return aggregated


def perform_statistical_tests(
    aggregated: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform statistical tests comparing network types.
    
    Args:
        aggregated: Aggregated metrics by network type
        
    Returns:
        Dictionary with test results
    """
    tests = {
        "pairwise_ttests": {},
        "anova": None,
    }
    
    try:
        from scipy import stats
        
        # Get all win rates
        all_data = {}
        for network_type in NETWORK_TYPES:
            if "final_win_rates" in aggregated.get(network_type, {}):
                all_data[network_type] = aggregated[network_type]["final_win_rates"]
        
        if len(all_data) < 2:
            return tests
        
        # Pairwise t-tests with Bonferroni correction
        n_comparisons = len(NETWORK_TYPES) * (len(NETWORK_TYPES) - 1) // 2
        alpha_bonferroni = 0.05 / n_comparisons
        
        for i, t1 in enumerate(NETWORK_TYPES):
            for t2 in NETWORK_TYPES[i+1:]:
                if t1 in all_data and t2 in all_data:
                    # Check normality
                    _, p_norm1 = stats.shapiro(all_data[t1]) if len(all_data[t1]) >= 3 else (None, 1.0)
                    _, p_norm2 = stats.shapiro(all_data[t2]) if len(all_data[t2]) >= 3 else (None, 1.0)
                    
                    # Check equal variance
                    _, p_levene = stats.levene(all_data[t1], all_data[t2])
                    
                    # Use appropriate test
                    if p_norm1 > 0.05 and p_norm2 > 0.05:
                        # Parametric t-test
                        if p_levene > 0.05:
                            t_stat, p_value = stats.ttest_ind(all_data[t1], all_data[t2])
                            test_used = "t-test (equal variance)"
                        else:
                            t_stat, p_value = stats.ttest_ind(all_data[t1], all_data[t2], equal_var=False)
                            test_used = "Welch's t-test"
                    else:
                        # Non-parametric Mann-Whitney U
                        t_stat, p_value = stats.mannwhitneyu(all_data[t1], all_data[t2], alternative='two-sided')
                        test_used = "Mann-Whitney U"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(all_data[t1], ddof=1)**2 + np.std(all_data[t2], ddof=1)**2) / 2)
                    cohens_d = (np.mean(all_data[t1]) - np.mean(all_data[t2])) / pooled_std if pooled_std > 0 else 0
                    
                    tests["pairwise_ttests"][f"{t1}_vs_{t2}"] = {
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "p_value_bonferroni": float(p_value * n_comparisons),
                        "significant_raw": p_value < 0.05,
                        "significant_bonferroni": p_value < alpha_bonferroni,
                        "test_used": test_used,
                        "cohens_d": float(cohens_d),
                        "effect_interpretation": interpret_cohens_d(cohens_d),
                    }
        
        # One-way ANOVA (if all types have data)
        if len(all_data) == len(NETWORK_TYPES):
            groups = [all_data[t] for t in NETWORK_TYPES]
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Effect size (eta-squared)
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_values - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            tests["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "eta_squared": float(eta_squared),
            }
    
    except ImportError:
        print("Warning: scipy not available for statistical tests")
    
    return tests


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def generate_comparison_bar_chart(
    aggregated: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a bar chart comparing win rates across network types.
    
    Args:
        aggregated: Aggregated metrics by network type
        output_path: Path to save the figure
    """
    import seaborn as sns
    
    # Prepare data
    types = []
    means = []
    errors = []
    
    for network_type in NETWORK_TYPES:
        if "mean" in aggregated.get(network_type, {}):
            types.append(network_type.replace("_", " ").title())
            means.append(aggregated[network_type]["mean"])
            # Use 95% CI for error bars
            ci_lower = aggregated[network_type]["ci_95_lower"]
            ci_upper = aggregated[network_type]["ci_95_upper"]
            errors.append([
                aggregated[network_type]["mean"] - ci_lower,
                ci_upper - aggregated[network_type]["mean"],
            ])
    
    if not types:
        print("No data available for bar chart")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(types))
    
    # Create bars
    x = np.arange(len(types))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    error_array = np.array(errors).T
    ax.errorbar(x, means, yerr=error_array, fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    # Add individual data points
    for i, network_type in enumerate(NETWORK_TYPES):
        if "final_win_rates" in aggregated.get(network_type, {}):
            win_rates = aggregated[network_type]["final_win_rates"]
            jitter = np.random.uniform(-0.15, 0.15, len(win_rates))
            ax.scatter(
                [i + j for j in jitter],
                win_rates,
                color='darkgray',
                s=50,
                alpha=0.6,
                zorder=5,
                edgecolor='black',
                linewidth=0.5,
            )
    
    # Customize
    ax.set_ylabel("Win Rate", fontsize=14)
    ax.set_xlabel("Network Type", fontsize=14)
    ax.set_title("Final Win Rate by Network Type\n(Mean ± 95% CI, with individual runs)", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{mean:.1%}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated: {output_path}")


def generate_training_curves(
    aggregated: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate training curves showing win rate over rounds with confidence bands.
    
    Args:
        aggregated: Aggregated metrics by network type
        output_path: Path to save the figure
    """
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(NETWORK_TYPES))
    
    for idx, network_type in enumerate(NETWORK_TYPES):
        if "round_win_rates" not in aggregated.get(network_type, {}):
            continue
        
        round_data = aggregated[network_type]["round_win_rates"]
        if not round_data:
            continue
        
        # Align all runs to same length (use min length)
        min_length = min(len(r) for r in round_data)
        aligned_data = np.array([r[:min_length] for r in round_data])
        
        rounds = np.arange(1, min_length + 1)
        mean_wr = np.mean(aligned_data, axis=0)
        std_wr = np.std(aligned_data, axis=0, ddof=1)
        
        # Plot mean line
        label = network_type.replace("_", " ").title()
        ax.plot(rounds, mean_wr, linewidth=2.5, label=label, color=colors[idx], zorder=3)
        
        # Plot confidence band (mean ± std)
        ax.fill_between(
            rounds,
            mean_wr - std_wr,
            mean_wr + std_wr,
            alpha=0.2,
            color=colors[idx],
        )
        
        # Plot individual runs with low opacity
        for run_data in round_data:
            ax.plot(
                range(1, len(run_data) + 1),
                run_data,
                alpha=0.15,
                linewidth=1,
                color=colors[idx],
            )
    
    ax.set_xlabel("Training Round", fontsize=14)
    ax.set_ylabel("Win Rate", fontsize=14)
    ax.set_title("Training Progress by Network Type\n(Mean ± 1 Std Dev, individual runs shown)", fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated: {output_path}")


def generate_box_plot(
    aggregated: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a box plot comparing win rate distributions.
    
    Args:
        aggregated: Aggregated metrics by network type
        output_path: Path to save the figure
    """
    import seaborn as sns
    
    # Prepare data
    data = []
    labels = []
    
    for network_type in NETWORK_TYPES:
        if "final_win_rates" in aggregated.get(network_type, {}):
            win_rates = aggregated[network_type]["final_win_rates"]
            data.extend(win_rates)
            labels.extend([network_type.replace("_", " ").title()] * len(win_rates))
    
    if not data:
        print("No data available for box plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create box plot with individual points
    import pandas as pd
    df = pd.DataFrame({"Network Type": labels, "Win Rate": data})
    
    sns.boxplot(
        x="Network Type",
        y="Win Rate",
        data=df,
        ax=ax,
        palette="husl",
        width=0.5,
    )
    
    sns.stripplot(
        x="Network Type",
        y="Win Rate",
        data=df,
        ax=ax,
        color="black",
        alpha=0.5,
        size=8,
        jitter=0.1,
    )
    
    ax.set_ylabel("Win Rate", fontsize=14)
    ax.set_xlabel("Network Type", fontsize=14)
    ax.set_title("Win Rate Distribution by Network Type", fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated: {output_path}")


def print_results_table(
    aggregated: Dict[str, Dict[str, Any]],
    tests: Dict[str, Any],
) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Network Type':<15} {'N':>4} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'Range':>15}")
    print("-" * 80)
    
    for network_type in NETWORK_TYPES:
        if "mean" in aggregated.get(network_type, {}):
            s = aggregated[network_type]
            ci_str = f"[{s['ci_95_lower']:.1%}, {s['ci_95_upper']:.1%}]"
            range_str = f"[{s['min']:.1%}, {s['max']:.1%}]"
            print(f"{network_type:<15} {s['n_runs']:>4} {s['mean']:>7.1%} {s['std']:>7.1%} {ci_str:>20} {range_str:>15}")
        else:
            error = aggregated.get(network_type, {}).get("error", "No data")
            print(f"{network_type:<15} {error}")
    
    # Print statistical tests
    if tests.get("pairwise_ttests"):
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISONS")
        print(f"{'='*80}")
        
        for comparison, result in tests["pairwise_ttests"].items():
            sig = ""
            if result["significant_bonferroni"]:
                sig = "***"
            elif result["significant_raw"]:
                sig = "*"
            
            print(f"\n{comparison.replace('_', ' ')}:")
            print(f"  Test: {result['test_used']}")
            print(f"  p-value: {result['p_value']:.4f} {sig}")
            print(f"  p-value (Bonferroni): {result['p_value_bonferroni']:.4f}")
            print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_interpretation']})")
    
    if tests.get("anova"):
        print(f"\nANOVA (all groups):")
        print(f"  F-statistic: {tests['anova']['f_statistic']:.3f}")
        print(f"  p-value: {tests['anova']['p_value']:.4f}")
        print(f"  η²: {tests['anova']['eta_squared']:.3f}")
    
    print(f"\n* p < 0.05 (uncorrected), *** p < 0.05 (Bonferroni corrected)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Generate comparison graphs"
    )
    parser.add_argument(
        "--export",
        choices=["json", "csv", "latex"],
        help="Export results in specified format"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for graphs and exports"
    )
    
    args = parser.parse_args()
    
    # Get current experiment
    manager = ExperimentManager.get_current()
    if not manager:
        print("No current experiment. Run 'experiment_manager.py init' first.")
        return 1
    
    # Aggregate metrics
    print("Aggregating metrics from all runs...")
    aggregated = aggregate_metrics_by_type(manager)
    
    # Perform statistical tests
    print("Performing statistical tests...")
    tests = perform_statistical_tests(aggregated)
    
    # Print results
    print_results_table(aggregated, tests)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = manager.experiment_path / "analysis" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate graphs
    if args.graphs:
        if not HAS_MATPLOTLIB:
            print("\nWarning: matplotlib not installed. Skipping graph generation.")
            print("Install with: pip install matplotlib seaborn")
        else:
            print("\nGenerating comparison graphs...")
            generate_comparison_bar_chart(aggregated, output_dir / "win_rate_comparison.png")
            generate_training_curves(aggregated, output_dir / "training_curves.png")
            generate_box_plot(aggregated, output_dir / "win_rate_distribution.png")
    
    # Export results
    if args.export:
        export_path = output_dir / f"results.{args.export}"
        
        if args.export == "json":
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "experiment": manager.metadata["experiment_name"],
                "aggregated_results": {k: {**v, "final_win_rates": list(v.get("final_win_rates", []))} 
                                       for k, v in aggregated.items()},
                "statistical_tests": tests,
            }
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif args.export == "csv":
            import csv
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Network Type", "N", "Mean", "Std", "CI Lower", "CI Upper", "Min", "Max"])
                for network_type in NETWORK_TYPES:
                    if "mean" in aggregated.get(network_type, {}):
                        s = aggregated[network_type]
                        writer.writerow([
                            network_type, s["n_runs"], s["mean"], s["std"],
                            s["ci_95_lower"], s["ci_95_upper"], s["min"], s["max"]
                        ])
        
        elif args.export == "latex":
            with open(export_path, 'w') as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Win Rate by Network Type}\n")
                f.write("\\label{tab:win_rates}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Network Type & N & Mean & Std & 95\\% CI \\\\\n")
                f.write("\\hline\n")
                
                for network_type in NETWORK_TYPES:
                    if "mean" in aggregated.get(network_type, {}):
                        s = aggregated[network_type]
                        name = network_type.replace("_", " ").title()
                        ci = f"[{s['ci_95_lower']:.1%}, {s['ci_95_upper']:.1%}]"
                        f.write(f"{name} & {s['n_runs']} & {s['mean']:.1%} & {s['std']:.1%} & {ci} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        
        print(f"\nExported to: {export_path}")
    
    # Save full analysis
    analysis_file = manager.experiment_path / "analysis" / "full_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "aggregated_results": {k: {**v, "final_win_rates": list(v.get("final_win_rates", [])),
                                       "round_win_rates": [[float(x) for x in r] for r in v.get("round_win_rates", [])]} 
                                   for k, v in aggregated.items()},
            "statistical_tests": tests,
        }, f, indent=2)
    
    print(f"\nFull analysis saved to: {analysis_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
