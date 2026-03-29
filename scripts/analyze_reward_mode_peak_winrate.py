#!/usr/bin/env python3
"""
Analyze experiment logs in experiments/ and ThesisExpStroage/ to find which
reward_mode has the highest peak win rate vs GapMaximizer.

Validation is split into agent-as-P1 and agent-as-P2; we use the average
win rate across both seats per round, then take the peak (max over rounds) per run.

Results are split by algorithm (DQN vs PPO). Experiments without reward_mode
in config are assumed to use the project's current reward structure (hyperparams_config.json).
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIRS = [
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "ThesisExpStroage",
]

# Current active reward mode (used when experiment config has no reward_mode)
CURRENT_REWARD_MODE = "binary"


def get_current_reward_mode() -> str:
    """Read current reward_mode from project hyperparams_config.json."""
    config_path = PROJECT_ROOT / "hyperparams_config.json"
    if not config_path.exists():
        return CURRENT_REWARD_MODE
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("training", {}).get("reward_mode", CURRENT_REWARD_MODE)
    except (json.JSONDecodeError, OSError):
        return CURRENT_REWARD_MODE


def get_algorithm(experiment_path: Path) -> str:
    """Get algorithm (dqn or ppo) from experiment's base_hyperparams_config.json."""
    config_path = experiment_path / "base_hyperparams_config.json"
    if not config_path.exists():
        return "dqn"
    try:
        with open(config_path) as f:
            config = json.load(f)
        algo = config.get("algorithm", "dqn")
        return algo if algo in ("dqn", "ppo") else "dqn"
    except (json.JSONDecodeError, OSError):
        return "dqn"


def get_reward_mode(experiment_path: Path, current_active: str) -> str:
    """Get reward_mode from experiment's base_hyperparams_config.json; use current_active if missing."""
    config_path = experiment_path / "base_hyperparams_config.json"
    if not config_path.exists():
        return current_active
    try:
        with open(config_path) as f:
            config = json.load(f)
        mode = config.get("training", {}).get("reward_mode")
        return mode if mode is not None else current_active
    except (json.JSONDecodeError, OSError):
        return current_active


def get_use_double_dqn(experiment_path: Path) -> bool:
    """Get use_double_dqn from experiment's base_hyperparams_config.json (top-level; DQN only)."""
    config_path = experiment_path / "base_hyperparams_config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            config = json.load(f)
        return bool(config.get("use_double_dqn", False))
    except (json.JSONDecodeError, OSError):
        return False


def get_metrics_logs_dir(run_path: Path) -> Optional[Path]:
    """Return path to metrics_logs for this run (run_path or run_path/workspace)."""
    direct = run_path / "metrics_logs"
    if direct.exists():
        return direct
    workspace = run_path / "workspace" / "metrics_logs"
    if workspace.exists():
        return workspace
    return None


def extract_round_win_rate(metrics_logs_dir: Path, round_num: int) -> Optional[float]:
    """
    For a given round, get average trainee win rate vs GapMaximizer across both seats when
    available; otherwise use single-file (legacy) format.
    - New: trainee_first -> p1_win_rate, trainee_second -> p2_win_rate; average = (p1+p2)/2.
    - Old: single metrics_round_N_vs_gapmaximizer.jsonl; trainee is PlayerAgent -> use that side's win_rate.
    """
    first_file = metrics_logs_dir / f"metrics_round_{round_num}_vs_gapmaximizer_trainee_first.jsonl"
    second_file = metrics_logs_dir / f"metrics_round_{round_num}_vs_gapmaximizer_trainee_second.jsonl"
    if first_file.exists() and second_file.exists():
        rates = []
        for path, key in [(first_file, "p1_win_rate"), (second_file, "p2_win_rate")]:
            try:
                with open(path) as f:
                    lines = f.readlines()
                if not lines:
                    return None
                data = json.loads(lines[-1].strip())
                rates.append(data.get(key, 0.0))
            except (json.JSONDecodeError, OSError):
                return None
        if len(rates) != 2:
            return None
        return (rates[0] + rates[1]) / 2.0
    # Legacy: single file per round (e.g. metrics_round_N_vs_gapmaximizer.jsonl)
    single_file = metrics_logs_dir / f"metrics_round_{round_num}_vs_gapmaximizer.jsonl"
    if not single_file.exists():
        return None
    try:
        with open(single_file) as f:
            lines = f.readlines()
        if not lines:
            return None
        data = json.loads(lines[-1].strip())
        p1_name = data.get("p1_name", "")
        p2_name = data.get("p2_name", "")
        if "PlayerAgent" in p1_name or p1_name == "PlayerAgent":
            return data.get("p1_win_rate", 0.0)
        if "PlayerAgent" in p2_name or p2_name == "PlayerAgent":
            return data.get("p2_win_rate", 0.0)
        return None
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def get_round_numbers(metrics_logs_dir: Path) -> list[int]:
    """Get all round numbers that have gapmaximizer validation (both seats or legacy single file)."""
    rounds = set()
    for f in metrics_logs_dir.glob("metrics_round_*_vs_gapmaximizer*.jsonl"):
        if "selfplay" in f.name:
            continue
        stem = f.stem
        if "_vs_gapmaximizer" not in stem:
            continue
        parts = stem.split("_")
        if len(parts) >= 3 and parts[1] == "round":
            try:
                rounds.add(int(parts[2]))
            except ValueError:
                pass
    result = []
    for r in rounds:
        first = metrics_logs_dir / f"metrics_round_{r}_vs_gapmaximizer_trainee_first.jsonl"
        second = metrics_logs_dir / f"metrics_round_{r}_vs_gapmaximizer_trainee_second.jsonl"
        single = metrics_logs_dir / f"metrics_round_{r}_vs_gapmaximizer.jsonl"
        if first.exists() and second.exists():
            result.append(r)
        elif single.exists():
            result.append(r)
    return sorted(result)


def peak_win_rate_for_run(run_path: Path) -> tuple[Optional[float], Optional[int]]:
    """
    Compute peak (over rounds) average win rate vs GapMaximizer for this run.
    Returns (peak_win_rate, best_round) or (None, None) if no data.
    """
    metrics_logs_dir = get_metrics_logs_dir(run_path)
    if not metrics_logs_dir:
        return None, None
    round_nums = get_round_numbers(metrics_logs_dir)
    if not round_nums:
        return None, None
    best_rate = -1.0
    best_round = None
    for r in round_nums:
        rate = extract_round_win_rate(metrics_logs_dir, r)
        if rate is not None and rate > best_rate:
            best_rate = rate
            best_round = r
    if best_round is None:
        return None, None
    return best_rate, best_round


def main():
    # (algorithm, reward_mode) -> list of (peak_win_rate, experiment_name, run_id)
    by_algo_and_mode: dict[Tuple[str, str], list[tuple[float, str, str]]] = defaultdict(list)
    # DQN only: Double DQN vs normal DQN for comparison
    dqn_double_dqn: list[tuple[float, str, str]] = []
    dqn_normal_dqn: list[tuple[float, str, str]] = []
    current_active = get_current_reward_mode()
    runs_processed = 0
    runs_with_gap = 0

    for experiments_dir in EXPERIMENTS_DIRS:
        if not experiments_dir.exists():
            continue
        for exp_path in sorted(experiments_dir.iterdir()):
            if not exp_path.is_dir() or not exp_path.name.startswith("experiment_"):
                continue
            algorithm = get_algorithm(exp_path)
            reward_mode = get_reward_mode(exp_path, current_active)
            use_double = get_use_double_dqn(exp_path)
            runs_dir = exp_path / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                runs_processed += 1
                peak, best_round = peak_win_rate_for_run(run_dir)
                if peak is None:
                    continue
                runs_with_gap += 1
                key = (algorithm, reward_mode)
                entry = (peak, exp_path.name, run_dir.name)
                by_algo_and_mode[key].append(entry)
                if algorithm == "dqn":
                    if use_double:
                        dqn_double_dqn.append(entry)
                    else:
                        dqn_normal_dqn.append(entry)

    print("Reward mode vs GapMaximizer — peak average win rate (across both seats)")
    print("(Peak = max over validation rounds; each round = (P1_win_rate + P2_win_rate)/2)")
    print("(Experiments without reward_mode in config counted as current active: {!r})".format(current_active))
    print()

    if not by_algo_and_mode:
        print("No runs with GapMaximizer validation metrics found.")
        return

    def report_section(algorithm: str, label: str) -> None:
        entries_by_mode: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        for (algo, mode), entries in by_algo_and_mode.items():
            if algo != algorithm:
                continue
            entries_by_mode[mode].extend(entries)
        if not entries_by_mode:
            print(f"  No {label} runs with GapMaximizer validation.")
            return
        best_peak = -1.0
        best_mode = None
        best_run_info = None
        print(f"  --- {label} ---")
        for mode in sorted(entries_by_mode.keys()):
            entries = entries_by_mode[mode]
            peaks = [e[0] for e in entries]
            max_peak = max(peaks)
            mean_peak = sum(peaks) / len(peaks)
            top = max(entries, key=lambda x: x[0])
            print(f"    reward_mode = {mode!r}")
            print(f"      runs: {len(entries)}, max peak: {max_peak:.2%}, mean peak: {mean_peak:.2%}")
            print(f"      best: {top[2]} in {top[1]} -> {top[0]:.2%}")
            if max_peak > best_peak:
                best_peak = max_peak
                best_mode = mode
                best_run_info = top
        print(f"  {label} best peak: {best_peak:.2%} ({best_mode!r}) — {best_run_info[2]} in {best_run_info[1]}")
        print()

    report_section("ppo", "PPO")
    report_section("dqn", "DQN")

    # Double DQN vs normal DQN (peak vs GapMaximizer)
    print("  --- Double DQN vs normal DQN ---")
    for label, entries in [("Double DQN", dqn_double_dqn), ("Normal DQN", dqn_normal_dqn)]:
        if not entries:
            print(f"    {label}: no runs")
            continue
        peaks = [e[0] for e in entries]
        max_peak = max(peaks)
        mean_peak = sum(peaks) / len(peaks)
        top = max(entries, key=lambda x: x[0])
        print(f"    {label}: runs={len(entries)}, max peak={max_peak:.2%}, mean peak={mean_peak:.2%}")
        print(f"      best: {top[2]} in {top[1]} -> {top[0]:.2%}")
    if dqn_double_dqn and dqn_normal_dqn:
        best_double = max(e[0] for e in dqn_double_dqn)
        best_normal = max(e[0] for e in dqn_normal_dqn)
        winner = "Double DQN" if best_double > best_normal else "Normal DQN"
        print(f"    => Higher peak: {winner} ({max(best_double, best_normal):.2%} vs {min(best_double, best_normal):.2%})")
    print()

    # Global best
    all_entries = []
    for entries in by_algo_and_mode.values():
        all_entries.extend(entries)
    best = max(all_entries, key=lambda x: x[0])
    print("=" * 60)
    print(f"Overall highest peak: {best[0]:.2%} — {best[2]} in {best[1]}")
    print("=" * 60)
    print(f"(Processed {runs_processed} runs, {runs_with_gap} had GapMaximizer validation.)")


if __name__ == "__main__":
    main()
