"""
Generate stylized graph images from training metrics (post-training).
Uses the same config and data handling as the live dashboard; reads all
metrics once and writes one image per stat for reports/papers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt

from config_loader import ConfigError, load_config
from metrics_reader import (
    MetricsReader,
    TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
    TRANEE_WIN_RATE_VS_RANDOMIZED_KEY,
    is_training_only_stat,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "stats_dashboard.json"

# No limit when exporting (we want full history)
EXPORT_MAX_POINTS = 10_000_000

# Styling: readable fonts, grid, consistent colors
FIGURE_DPI = 150
FIGURE_SIZE = (8, 4)
FONT_SIZE_TITLE = 12
FONT_SIZE_LABELS = 11
FONT_SIZE_TICKS = 10
LINE_WIDTH = 1.8
GRID_ALPHA = 0.4
PRIMARY_COLOR = "#2563eb"   # blue
SECONDARY_COLOR = "#059669" # teal (for scatter/bar accent)
# Rolling average for noisy metrics (same-graph overlay)
# Episode-based traces usually benefit from heavier smoothing.
ROLLING_WINDOW_EPISODE = 500
# Round-based win-rate traces are shorter; keep them more responsive.
ROLLING_WINDOW_ROUND = 5
RAW_ALPHA = 0.25  # transparency for raw series when rolling avg is shown
ROLLING_LINE_WIDTH = 2.0   # slightly thicker so trend stands out
STD_FILL_ALPHA = 0.1  # transparency for ±1 std shaded region
# Colors for multiple runs (one per metrics folder)
RUN_COLORS = [
    "#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#0d9488",
]


def _rolling_average(x: list[float], y: list[float], window: int) -> tuple[list[float], list[float]]:
    """Compute rolling mean of y (same length); x is returned as-is for alignment."""
    if not y or window < 1:
        return (list(x), list(y))
    n = len(y)
    out_y: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        out_y.append(sum(chunk) / len(chunk))
    return (list(x), out_y)


def _rolling_std(y: list[float], window: int) -> list[float]:
    """Compute rolling standard deviation of y (same length as y)."""
    if not y or window < 1:
        return list(y) if y else []
    n = len(y)
    out: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        m = sum(chunk) / len(chunk)
        variance = sum((v - m) ** 2 for v in chunk) / len(chunk)
        out.append(variance ** 0.5)
    return out


def _x_label(stat_key: str) -> str:
    if stat_key in (
        TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
        TRANEE_WIN_RATE_VS_RANDOMIZED_KEY,
    ):
        return "Round"
    if is_training_only_stat(stat_key):
        return "Training episode"
    return "Episode"


def _rolling_window_for_stat(stat_key: str) -> int:
    """Choose smoothing window based on metric x-axis semantics."""
    if stat_key in (
        TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
        TRANEE_WIN_RATE_VS_RANDOMIZED_KEY,
    ):
        return ROLLING_WINDOW_ROUND
    return ROLLING_WINDOW_EPISODE


def _plot_series(
    ax,
    stat_key: str,
    label: str,
    graph_type: str,
    x: list[float],
    y: list[float],
    *,
    alpha: float = 1.0,
    color: str | None = None,
) -> None:
    if not x or not y:
        return
    c = color or PRIMARY_COLOR
    if graph_type == "line":
        # Raw series are rendered as dashed to distinguish them from rolling averages.
        ax.plot(x, y, color=c, linewidth=LINE_WIDTH, alpha=alpha, label=label, linestyle="--")
    elif graph_type == "scatter":
        ax.scatter(x, y, color=c, s=8, alpha=min(alpha, 0.7), label=label)
    elif graph_type == "bar":
        ax.bar(x, y, color=c, alpha=0.8, width=0.8, label=label)
    elif graph_type == "histogram":
        ax.hist(y, bins=min(50, max(10, len(y) // 5)), color=c, alpha=0.8, edgecolor="white", linewidth=0.3, label=label)
    else:
        ax.plot(x, y, color=c, linewidth=LINE_WIDTH, alpha=alpha, label=label, linestyle="--")


def _legend_run_name(metrics_arg_index: int) -> str:
    """Legend label for a run: 1-based index matching --metrics order."""
    return f"Run {metrics_arg_index + 1}"


def _infer_architecture_from_metrics_paths(paths: list[Path]) -> str | None:
    """
    Guess architecture label from path strings (run folders often contain scale_11 / large_hidden).
    Returns None if ambiguous or unknown.
    """
    blob = " ".join(str(p.resolve()) for p in paths).lower()
    scale_markers = ("scale_11", "scale11", "ppo_scale11", "ppos11")
    lh_markers = ("large_hidden", "ppo_large_hidden", "ppolh")
    has_scale = any(m in blob for m in scale_markers)
    has_lh = any(m in blob for m in lh_markers)
    if has_scale and not has_lh:
        return "Scale_11"
    if has_lh and not has_scale:
        return "Large_hidden"
    return None


def _format_figure_title(architecture: str | None, label: str) -> str:
    """Prefix title as 'Architecture : label' when architecture is set."""
    if architecture and architecture.strip():
        return f"{architecture.strip()} : {label}"
    return label


def _save_figure(
    out_dir: Path,
    stat_key: str,
    label: str,
    graph_type: str,
    series_list: list[tuple[int, list[float], list[float]]],
    formats: list[str],
    dpi: int,
    y_label: str | None = None,
    architecture: str | None = None,
) -> list[Path]:
    """Build one figure per stat and save in requested formats. series_list = [(metrics_arg_index, x, y), ...]."""
    written: list[Path] = []
    # Drop runs with no data
    series_list = [(arg_idx, x, y) for arg_idx, x, y in series_list if x and y]
    if not series_list:
        return written

    # Safe filename from stat key
    base_name = stat_key.replace("/", "_").replace("\\", "_").strip() or "unnamed"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    if graph_type in ("line", "scatter"):
        single_run = len(series_list) == 1
        rolling_window = _rolling_window_for_stat(stat_key)
        for plot_idx, (arg_idx, x, y) in enumerate(series_list):
            color = PRIMARY_COLOR if single_run else RUN_COLORS[plot_idx % len(RUN_COLORS)]
            raw_label = "Raw" if single_run else None
            _plot_series(ax, stat_key, raw_label, graph_type, x, y, alpha=RAW_ALPHA, color=color)
            x_roll, y_roll = _rolling_average(x, y, rolling_window)
            std_roll = _rolling_std(y, rolling_window)
            if single_run:
                ax.fill_between(
                    x_roll,
                    [a - s for a, s in zip(y_roll, std_roll)],
                    [a + s for a, s in zip(y_roll, std_roll)],
                    color=color,
                    alpha=STD_FILL_ALPHA,
                    label=f"±1 std ({rolling_window})",
                )
            else:
                ax.fill_between(
                    x_roll,
                    [a - s for a, s in zip(y_roll, std_roll)],
                    [a + s for a, s in zip(y_roll, std_roll)],
                    color=color,
                    alpha=STD_FILL_ALPHA,
                )
            line_label = _legend_run_name(arg_idx)
            ax.plot(
                x_roll,
                y_roll,
                color=color,
                linewidth=ROLLING_LINE_WIDTH,
                label=line_label,
            )
    else:
        for plot_idx, (arg_idx, x, y) in enumerate(series_list):
            color = RUN_COLORS[plot_idx % len(RUN_COLORS)]
            _plot_series(ax, stat_key, _legend_run_name(arg_idx), graph_type, x, y, color=color)

    y_axis_label = y_label if y_label else label
    if graph_type == "histogram":
        ax.set_xlabel(label, fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel("Count", fontsize=FONT_SIZE_LABELS)
    else:
        ax.set_xlabel(_x_label(stat_key), fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel(y_axis_label, fontsize=FONT_SIZE_LABELS)
    ax.set_title(_format_figure_title(architecture, label), fontsize=FONT_SIZE_TITLE, fontweight="medium")
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="-")
    ax.legend(loc="best", fontsize=FONT_SIZE_TICKS)

    plt.tight_layout()

    for fmt in formats:
        ext = fmt.lower()
        if ext not in ("png", "svg", "pdf"):
            continue
        path = out_dir / f"{base_name}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight", format=ext)
        written.append(path)

    plt.close(fig)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training metrics as stylized graph images (PNG/SVG)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stats_dashboard.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more paths to metrics JSONL file(s) or directory(ies) containing *.jsonl. Each path is one run; all runs are overlaid on the same graph per stat.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exported_graphs"),
        help="Output directory for image files",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf", "png,svg", "png,pdf", "svg,pdf", "all"],
        default="png",
        help="Output format(s). 'all' = png, svg, pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help="DPI for PNG output (default %(default)s)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Short architecture label prepended to every figure title as 'NAME : <stat title>'. "
            "If omitted, a best-effort guess is made from --metrics paths "
            "(e.g. scale_11 / ppos11 → Scale_11, large_hidden / ppolh → Large_hidden)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}")
        return

    metrics_paths = [Path(p) for p in args.metrics]
    for p in metrics_paths:
        if not p.exists():
            print(f"Metrics path does not exist: {p}")
            return

    if args.format == "all":
        formats = ["png", "svg", "pdf"]
    else:
        formats = [s.strip() for s in args.format.split(",")]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    architecture = args.architecture
    if architecture is None:
        architecture = _infer_architecture_from_metrics_paths(metrics_paths)

    # Load each metrics path into its own reader; aggregate by stat as (arg_idx, x, y) per stat
    stat_keys = config.stat_keys()
    series_by_stat: dict[str, list[tuple[int, list[float], list[float]]]] = {
        k: [] for k in stat_keys
    }
    for arg_idx, metrics_path in enumerate(metrics_paths):
        reader = MetricsReader(stat_keys=stat_keys, max_points=EXPORT_MAX_POINTS)
        reader.add_path(metrics_path)
        reader.poll()
        for key in stat_keys:
            x, y = reader.get_series(key)
            series_by_stat[key].append((arg_idx, x, y))

    graph_types = {s.key: s.graph_type for s in config.stats}
    written: list[Path] = []
    for s in config.stats:
        series_list = series_by_stat.get(s.key, [])
        paths = _save_figure(
            out_dir,
            s.key,
            s.label,
            graph_types.get(s.key, "line"),
            series_list,
            formats,
            args.dpi,
            y_label=s.y_label,
            architecture=architecture,
        )
        written.extend(paths)

    print(f"Exported {len(written)} file(s) to {out_dir.resolve()}")
    for p in sorted(written):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
