#!/usr/bin/env python3
"""
Create an experiment matching the DQN conditions of the best ThesisExpStorage runs
(experiment_20260202_101706: large_hidden, self-play, 57% final / 63% peak vs GapMaximizer)
with extended validation (500 per position, 1000 extended).

Key settings from best runs (matches commit 4fe3e2a7519f):
- Self-play (train_vs_gapmaximizer=False)
- Large_hidden [512] and scale_11 game_based
- NO PER, NO Double DQN
- Binary reward
- use_position_indicator: false (old 468-dim fusion, no P1/P2 one-hot)
- eps_end 0.15, eps_decay 20000
- lr 5e-5, lr_decay_interval 5, lr_decay_rate 0.9
- replay_buffer_size 30000
- 10 rounds, 250 eps/round
- validation_opponent: both
- Extra validation: 500 per position, 1000 extended

Usage:
    python create_best_storage_match_experiment.py
    python create_best_storage_match_experiment.py --runs-per-arch 3
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

project_root = Path(__file__).resolve().parent.parent
experiments_dir = project_root / "experiments"

SCALE_11_HIDDEN_LAYERS = [572, 143, 165]
ROUNDS_PER_RUN = 20
EPS_PER_ROUND = 250
RUNS_PER_ARCHITECTURE = 2


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def generate_seeds(count: int, base_seed: int = 42) -> List[int]:
    np.random.seed(base_seed)
    return [int(np.random.randint(1, 100000)) for _ in range(count)]


def _run_template(
    run_id: str,
    run_number: int,
    seed: int,
    network_type: str,
    input_type: str = "embedding",
    scale: Optional[int] = None,
    hidden_layers: Optional[List[int]] = None,
) -> dict:
    return {
        "run_id": run_id,
        "network_type": network_type,
        "input_type": input_type,
        "scale": scale,
        "hidden_layers": hidden_layers or ([512] if network_type == "large_hidden" else SCALE_11_HIDDEN_LAYERS),
        "rounds": ROUNDS_PER_RUN,
        "run_number": run_number,
        "seed": seed,
        "status": "pending",
        "created_at": None,
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "final_metrics": None,
        "error_message": None,
    }


def create_best_storage_match_experiment(
    name: str = "best_storage_match",
    description: str = "",
    runs_per_arch: int = RUNS_PER_ARCHITECTURE,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_{name}"
    experiment_path = experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    (experiment_path / "runs").mkdir(exist_ok=True)

    total_runs = 2 * runs_per_arch
    seeds = generate_seeds(total_runs)

    runs = {}
    run_num = 0

    for i in range(runs_per_arch):
        run_num += 1
        run_id = f"scale_11_embedding_run_{i + 1:02d}"
        runs[run_id] = _run_template(
            run_id=run_id,
            run_number=run_num,
            seed=seeds[2 * i],
            network_type="game_based",
            input_type="embedding",
            scale=11,
            hidden_layers=SCALE_11_HIDDEN_LAYERS,
        )

    for i in range(runs_per_arch):
        run_num += 1
        run_id = f"large_hidden_embedding_run_{i + 1:02d}"
        runs[run_id] = _run_template(
            run_id=run_id,
            run_number=run_num,
            seed=seeds[2 * i + 1],
            network_type="large_hidden",
            input_type="embedding",
            scale=None,
            hidden_layers=[512],
        )

    run_order: List[str] = []
    for i in range(runs_per_arch):
        run_order.append(f"scale_11_embedding_run_{i + 1:02d}")
        run_order.append(f"large_hidden_embedding_run_{i + 1:02d}")

    base_desc = (
        f"Match best ThesisExpStorage DQN: self-play, large_hidden + scale_11, "
        f"no PER/Double DQN, binary reward. 10 rounds, 250 eps/round. "
        f"Extended validation: 500 per position, 1000 extended."
    )

    metadata = {
        "experiment_name": experiment_name,
        "display_name": name,
        "description": description or base_desc,
        "experiment_type": "best_storage_match",
        "train_vs_gapmaximizer": False,
        "validation_opponent": "both",
        "use_double_dqn": False,
        "use_prioritized_replay": False,
        "reward_mode": "binary",
        "rounds_per_run": ROUNDS_PER_RUN,
        "eps_per_round": EPS_PER_ROUND,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "run_order": run_order,
        "runs_per_architecture": runs_per_arch,
        "total_runs": total_runs,
        "seeds": seeds,
    }

    base_config_src = project_root / "hyperparams_config.json"
    base_config_dst = experiment_path / "base_hyperparams_config.json"
    shutil.copy(base_config_src, base_config_dst)
    with open(base_config_dst) as f:
        config = json.load(f)

    config["use_double_dqn"] = False
    config["use_prioritized_replay"] = False
    config["use_position_indicator"] = False  # Match old 468-dim network (no P1/P2 in obs)
    config["embedding_dim"] = 32  # Match old ThesisExpStorage config
    config["zone_encoded_dim"] = 52
    config["eps_decay"] = 20000
    config["eps_end"] = 0.15
    config["learning_rate"] = 5e-5
    config["lr_decay_interval"] = 5
    config["lr_decay_rate"] = 0.9
    config["replay_buffer_size"] = 30000

    if "training" not in config:
        config["training"] = {}
    config["training"]["train_vs_gapmaximizer"] = False
    config["training"]["validation_opponent"] = "both"
    config["training"]["reward_mode"] = "binary"
    config["training"]["rounds"] = ROUNDS_PER_RUN
    config["training"]["eps_per_round"] = EPS_PER_ROUND
    config["training"]["validation_episodes_per_position"] = 500
    config["training"]["extended_validation_episodes_per_position"] = 1000
    config["training"]["curriculum_vs_gapmaximizer_ratio"] = 0
    config["training"]["selfplay_historical_opponent_ratio"] = 0

    with open(base_config_dst, "w") as f:
        json.dump(config, f, indent=2)

    with open(experiment_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(experiment_path / "runs_status.json", "w") as f:
        json.dump(runs, f, indent=2)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / ".current_experiment", "w") as f:
        f.write(str(experiment_path))

    print(f"{'='*70}")
    print(f"Created experiment: {experiment_name}")
    print(f"{'='*70}\n")
    print(f"Location: {experiment_path}\n")
    print("DQN config (matches best ThesisExpStorage commit 4fe3e2a):")
    print("  - Self-play, validation vs both (Randomized + GapMaximizer)")
    print("  - No PER, no Double DQN, binary reward, use_position_indicator=False (468-dim)")
    print("  - eps_end: 0.15, eps_decay: 20000")
    print("  - lr: 5e-5, lr_decay_interval: 5, lr_decay_rate: 0.9")
    print(f"  - {ROUNDS_PER_RUN} rounds, {EPS_PER_ROUND} eps/round")
    print("  - Validation: 500 per position (extended 1000 every 5 rounds)")
    print(f"\nArchitectures: scale_11 (game_based), large_hidden [512]")
    print(f"Runs per architecture: {runs_per_arch} (total {total_runs} runs)")
    print(f"Run order:")
    for idx, rid in enumerate(run_order, 1):
        info = runs[rid]
        arch = "game_based " + str(info["hidden_layers"]) if info["network_type"] == "game_based" else "large_hidden [512]"
        print(f"  {idx}. {rid} — {arch}, seed {info['seed']}, {ROUNDS_PER_RUN} rounds")
    print(f"\nRun with: python experiment_manager.py run")
    print(f"{'='*70}\n")

    return experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create experiment matching best ThesisExpStorage DQN config with extended validation"
    )
    parser.add_argument("--name", "-n", default="best_storage_match", help="Experiment name")
    parser.add_argument("--description", "-d", default="", help="Experiment description")
    parser.add_argument("--runs-per-arch", "-r", type=int, default=RUNS_PER_ARCHITECTURE,
                        help=f"Runs per architecture (default: {RUNS_PER_ARCHITECTURE})")

    args = parser.parse_args()
    create_best_storage_match_experiment(
        args.name,
        args.description,
        runs_per_arch=args.runs_per_arch,
    )
