#!/usr/bin/env python3
"""
Generate DQN experiment graphs from ThesisExpStroage JSON (and optional JSONL metrics).

Standard DQN practice:
- Final performance: mean ± std over seeds (or multiple runs).
- Learning curves: validation win rate vs round number; optional smoothing over rounds.
- Grouping by condition (architecture, scale, input type) for comparison.

Usage:
  python scripts/plot_experiments.py [--storage PATH] [--out DIR] [--experiment NAME]
  python scripts/plot_experiments.py --list   # list experiments and exit
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = PROJECT_ROOT / "ThesisExpStroage"
DEFAULT_OUT = PROJECT_ROOT / "ThesisExpStroage" / "plots"


def discover_experiments(storage: Path) -> List[Path]:
    """Return list of experiment directories (those with runs_status.json)."""
    if not storage.is_dir():
        return []
    experiments = []
    for path in storage.iterdir():
        if path.is_dir() and (path / "runs_status.json").exists():
            experiments.append(path)
    return sorted(experiments, key=lambda p: p.name)


def load_runs_status(experiment_path: Path) -> Dict[str, Any]:
    """Load runs_status.json; return {} on missing/invalid."""
    p = experiment_path / "runs_status.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def load_experiment_metadata(experiment_path: Path) -> Dict[str, Any]:
    """Load experiment_metadata.json; return {} on missing/invalid."""
    p = experiment_path / "experiment_metadata.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def run_group_key(run_data: Dict[str, Any]) -> str:
    """Single string key for grouping runs (e.g. same architecture + scale + input)."""
    net = run_data.get("network_type", "unknown")
    inp = run_data.get("input_type", "unknown")
    scale = run_data.get("scale")
    if scale is not None:
        return f"{net}_{inp}_scale_{scale}"
    return f"{net}_{inp}"


def runs_to_dataframe(runs: Dict[str, Any]) -> pd.DataFrame:
    """Convert runs_status dict to a DataFrame with one row per run."""
    rows = []
    for run_id, data in runs.items():
        if not isinstance(data, dict):
            continue
        fm = data.get("final_metrics") or {}
        row = {
            "run_id": run_id,
            "network_type": data.get("network_type"),
            "input_type": data.get("input_type"),
            "scale": data.get("scale"),
            "seed": data.get("seed"),
            "status": data.get("status"),
            "final_win_rate": fm.get("final_win_rate"),
            "final_episode": fm.get("final_episode"),
            "duration_seconds": data.get("duration_seconds"),
            "group": run_group_key(data),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def filter_completed_with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only completed runs that have final_win_rate."""
    out = df[
        (df["status"] == "completed")
        & (df["final_win_rate"].notna())
    ].copy()
    return out


def plot_final_win_rate(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Bar chart: final win rate by group, mean ± std over runs (DQN standard).
    """
    if df.empty:
        return
    grouped = df.groupby("group", sort=False).agg(
        mean=("final_win_rate", "mean"),
        std=("final_win_rate", "std"),
        count=("final_win_rate", "count"),
    ).reset_index()
    # No std if single run
    grouped["std"] = grouped["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(grouped))
    bars = ax.bar(
        x,
        grouped["mean"],
        yerr=grouped["std"],
        capsize=5,
        color=plt.cm.Set2(np.linspace(0, 1, len(grouped))),
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["group"], rotation=45, ha="right")
    ax.set_ylabel("Final win rate")
    ax.set_xlabel("Condition (network_type_input_type_scale)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_win_rate_stripplot(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Strip plot: individual run win rates per group (shows distribution and seeds).
    """
    if df.empty:
        return
    groups = df["group"].unique().tolist()
    group_order = {g: i for i, g in enumerate(groups)}
    df = df.copy()
    df["x"] = df["group"].map(group_order)

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize)
    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        jitter = rng.uniform(-0.15, 0.15, size=len(sub)) if len(sub) > 1 else np.zeros(len(sub))
        ax.scatter(
            i + jitter,
            sub["final_win_rate"],
            alpha=0.8,
            s=60,
            label=g,
            edgecolors="black",
            linewidths=0.5,
        )
        if len(sub) > 1:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color="gray")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("Final win rate")
    ax.set_xlabel("Condition")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_jsonl_metrics(metrics_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSONL lines from metrics files in metrics_dir (e.g. validation win rate)."""
    if not metrics_dir.is_dir():
        return []
    records = []
    for f in sorted(metrics_dir.glob("*.jsonl")):
        # Prefer validation runs (vs randomized/gapmaximizer) for win rate
        try:
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
    return records


# Opponent names and trainee position as in metrics filenames: metrics_round_N_vs_{opponent}_{position}.jsonl
VALIDATION_OPPONENTS = ("randomized", "gapmaximizer")
TRAINEE_POSITIONS = ("trainee_first", "trainee_second")


def get_run_learning_curves_by_opponent_and_position(
    run_path: Path,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Extract (round_numbers, win_rates) per opponent and per trainee position.
    Keys: (opponent, position) e.g. ("randomized", "trainee_first").
    Uses end-of-round win rate (last line). trainee_first -> p1_win_rate, trainee_second -> p2_win_rate.
    """
    keys = [(opp, pos) for opp in VALIDATION_OPPONENTS for pos in TRAINEE_POSITIONS]
    out: Dict[Tuple[str, str], List[Tuple[int, float]]] = {k: [] for k in keys}
    metrics_dir = run_path / "metrics_logs"
    if not metrics_dir.is_dir():
        return {k: (np.array([]), np.array([])) for k in keys}
    pattern = re.compile(r"round_(\d+)_vs_(randomized|gapmaximizer)_(trainee_first|trainee_second)")
    for f in metrics_dir.glob("metrics_round_*_vs_*_trainee_*.jsonl"):
        m = pattern.search(f.name)
        if not m:
            continue
        round_num = int(m.group(1))
        opponent = m.group(2)
        position = m.group(3)
        key = (opponent, position)
        if key not in out:
            continue
        try:
            with open(f) as fp:
                lines = [L.strip() for L in fp if L.strip()]
            if not lines:
                continue
            last = json.loads(lines[-1])
            # trainee_first: trainee is P1 -> p1_win_rate; trainee_second: trainee is P2 -> p2_win_rate
            wr = last.get("p1_win_rate") if position == "trainee_first" else last.get("p2_win_rate")
            if wr is not None:
                out[key].append((round_num, wr))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    result = {}
    for key in keys:
        items = sorted(out[key], key=lambda x: x[0])
        if not items:
            result[key] = (np.array([]), np.array([]))
        else:
            result[key] = (np.array([x[0] for x in items]), np.array([x[1] for x in items]))
    return result


def _plot_one_opponent_curves(
    series_by_group: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    title: str,
    out_path: Path,
    smooth_window: Optional[int] = None,
    figsize: Tuple[float, float] = (9, 5),
) -> None:
    """Plot one line per group (mean across runs, or single run). Only two lines for two groups."""
    if not series_by_group:
        return
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series_by_group), 1)))
    for idx, (group, curves) in enumerate(series_by_group.items()):
        color = colors[idx % len(colors)]
        if len(curves) > 1:
            all_rounds = np.unique(np.concatenate([c[0] for c in curves]))
            interp_curves = []
            for rounds, wrs in curves:
                if len(rounds) < 2:
                    continue
                interp_curves.append(np.interp(all_rounds, rounds, wrs))
            if interp_curves:
                mean_wr = np.nanmean(interp_curves, axis=0)
                std_wr = np.nanstd(interp_curves, axis=0)
                if smooth_window and smooth_window > 1 and len(mean_wr) >= smooth_window:
                    mean_wr = pd.Series(mean_wr).rolling(smooth_window, min_periods=1).mean().values
                    std_wr = pd.Series(std_wr).rolling(smooth_window, min_periods=1).mean().values
                ax.plot(all_rounds, mean_wr, color=color, linewidth=2, label=group)
                ax.fill_between(all_rounds, mean_wr - std_wr, mean_wr + std_wr, color=color, alpha=0.25)
        else:
            rounds, wrs = curves[0]
            if smooth_window and smooth_window > 1 and len(wrs) >= smooth_window:
                wrs = pd.Series(wrs).rolling(smooth_window, min_periods=1).mean().values
            ax.plot(rounds, wrs, color=color, linewidth=2, label=group)
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation win rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_curves(
    experiment_path: Path,
    runs: Dict[str, Any],
    display_name: str,
    out_dir: Path,
    smooth_window: Optional[int] = None,
    figsize: Tuple[float, float] = (9, 5),
) -> None:
    """
    Plot validation win rate vs round for each opponent and each trainee position (first/second).
    Four plots: vs Randomized (trainee first), vs Randomized (trainee second),
    vs GapMaximizer (trainee first), vs GapMaximizer (trainee second).
    """
    runs_dir = experiment_path / "runs"
    if not runs_dir.is_dir():
        return
    keys = [(opp, pos) for opp in VALIDATION_OPPONENTS for pos in TRAINEE_POSITIONS]
    by_key: Dict[Tuple[str, str], Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {
        k: {} for k in keys
    }
    for run_id, data in runs.items():
        if data.get("status") != "completed" or not isinstance(data, dict):
            continue
        group = run_group_key(data)
        run_path = runs_dir / run_id
        curves = get_run_learning_curves_by_opponent_and_position(run_path)
        for key in keys:
            rounds, wrs = curves[key]
            if len(rounds) == 0:
                continue
            if group not in by_key[key]:
                by_key[key][group] = []
            by_key[key][group].append((rounds, wrs))

    opponent_labels = {"randomized": "Randomized", "gapmaximizer": "GapMaximizer"}
    position_labels = {"trainee_first": "trainee first", "trainee_second": "trainee second"}
    for (opp, position) in keys:
        series = by_key[(opp, position)]
        if not series:
            continue
        opp_label = opponent_labels.get(opp, opp)
        pos_label = position_labels.get(position, position)
        _plot_one_opponent_curves(
            series,
            title=f"Validation win rate vs round (vs {opp_label}, {pos_label}) — {display_name}",
            out_path=out_dir / f"learning_curves_vs_{opp}_{position}.png",
            smooth_window=smooth_window,
            figsize=figsize,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DQN experiment graphs from ThesisExpStroage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        help="Path to ThesisExpStroage (experiment root).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory to write plot images.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Process only this experiment dir name (e.g. experiment_20260202_113519_...). If not set, process all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments and exit.",
    )
    parser.add_argument(
        "--no-learning-curves",
        action="store_true",
        help="Skip learning curve plots (require metrics_logs JSONL).",
    )
    args = parser.parse_args()

    if args.list:
        experiments = discover_experiments(args.storage)
        print(f"Found {len(experiments)} experiments in {args.storage}")
        for exp in experiments:
            meta = load_experiment_metadata(exp)
            name = meta.get("display_name") or meta.get("experiment_name") or exp.name
            runs = load_runs_status(exp)
            completed = sum(1 for r in (runs or {}).values() if isinstance(r, dict) and r.get("status") == "completed" and (r.get("final_metrics") or {}).get("final_win_rate") is not None)
            print(f"  {exp.name}  ({completed} runs with final metrics)")
        return

    experiments = discover_experiments(args.storage)
    if args.experiment:
        experiments = [p for p in experiments if p.name == args.experiment]
        if not experiments:
            print(f"No experiment named '{args.experiment}' found.")
            return
    if not experiments:
        print(f"No experiments found in {args.storage}")
        return

    for exp_path in experiments:
        meta = load_experiment_metadata(exp_path)
        runs = load_runs_status(exp_path)
        display_name = meta.get("display_name") or meta.get("experiment_name") or exp_path.name
        safe_name = exp_path.name

        df = runs_to_dataframe(runs)
        df_plot = filter_completed_with_metrics(df)
        if df_plot.empty:
            print(f"[{safe_name}] No completed runs with final_win_rate; skipping plots.")
            continue

        out_dir = args.out / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        title = f"Final win rate — {display_name}"
        plot_final_win_rate(df_plot, title, out_dir / "final_win_rate_bars.png")
        plot_final_win_rate_stripplot(df_plot, title, out_dir / "final_win_rate_stripplot.png")
        print(f"[{safe_name}] Wrote final win rate plots to {out_dir}")

        if not args.no_learning_curves and runs:
            plot_learning_curves(exp_path, runs, display_name, out_dir)
            print(f"[{safe_name}] Wrote learning curve plots (4: per opponent × trainee first/second) if metrics_logs present.")

    print(f"Plots written to {args.out}")


if __name__ == "__main__":
    main()
