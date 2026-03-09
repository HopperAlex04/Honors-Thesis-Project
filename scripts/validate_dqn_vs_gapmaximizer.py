#!/usr/bin/env python3
"""
Extended validation run: load a DQN checkpoint and play many games vs GapMaximizer
to test if reported win rates are robust or due to variance.

The best ThesisExpStorage DQN (experiment_20260202_101706, large_hidden) reported
62.8% vs GapMaximizer at round 6. With only ~250-500 validation games per position,
that could have ~2–3% standard error. This script runs 5000 (default) games per
position for tighter confidence intervals.

Usage:
    python scripts/validate_dqn_vs_gapmaximizer.py --checkpoint path/to/model_best.pt
    python scripts/validate_dqn_vs_gapmaximizer.py --run experiments/exp/runs/large_hidden_run_01
    python scripts/validate_dqn_vs_gapmaximizer.py --checkpoint model.pt --episodes 10000

With --run: uses models/model_best.pt if present, else model_round_N.pt from best_round.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from cuttle import players as Players
from cuttle import training as Training
from cuttle.environment import CuttleEnvironment
from cuttle.networks import EmbeddingBasedNeuralNetwork, NeuralNetwork


def build_dqn_from_config(cfg, observation_space, num_actions, use_position_indicator: bool = True):
    """Build DQN model from config (matches train.build_dqn_model_from_config)."""
    net_type = cfg.get("network_type", "game_based")
    use_embeddings = cfg.get("use_embeddings", True)
    embedding_dim = cfg.get("embedding_dim", 52)
    zone_encoded_dim = cfg.get("zone_encoded_dim", 52)
    emb_size = cfg.get("embedding_size", 16)
    pos_ind = cfg.get("use_position_indicator", use_position_indicator)

    if use_embeddings:
        if net_type == "linear":
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[], use_position_indicator=pos_ind
            )
        if net_type == "large_hidden":
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=[512], use_position_indicator=pos_ind
            )
        if net_type == "game_based":
            scale = cfg.get("game_based_scale", 2)
            layers = cfg.get("game_based_hidden_layers")
            hidden = layers or [52 * scale, 13 * scale, 15 * scale]
            return EmbeddingBasedNeuralNetwork(
                observation_space, num_actions,
                embedding_dim=embedding_dim, zone_encoded_dim=zone_encoded_dim,
                hidden_layers=hidden, use_position_indicator=pos_ind
            )
    else:
        if net_type == "linear":
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[])
        if net_type == "large_hidden":
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[512])
        if net_type == "game_based":
            scale = cfg.get("game_based_scale", 2)
            layers = cfg.get("game_based_hidden_layers")
            hidden = layers or [52 * scale, 13 * scale, 15 * scale]
            return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=hidden)

    return NeuralNetwork(observation_space, emb_size, num_actions, None, hidden_layers=[52, 13, 15])


def load_dqn_agent(checkpoint_path: Path) -> Players.Agent:
    """Load a DQN agent from checkpoint. Acts greedily (eps=0)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})

    # Detect legacy checkpoints (trained before position indicator): fusion_dim 468 vs 470
    use_pos = True
    state = ckpt.get("model_state_dict", {})
    for k, v in state.items():
        if "hidden_layers.0.weight" in k and v.shape[1] == 468:
            use_pos = False
            break
        if "output_layer.weight" in k and v.shape[1] == 468:
            use_pos = False
            break
    cfg = {**cfg, "use_position_indicator": use_pos}

    env = CuttleEnvironment()
    actions = env.actions

    model = build_dqn_from_config(cfg, env.observation_space, actions)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch_size = cfg.get("batch_size", 128)
    gamma = cfg.get("gamma", 0.9)
    q_value_clip = cfg.get("q_value_clip", 15.0)
    use_double_dqn = cfg.get("use_double_dqn", False)

    trainee = Players.Agent(
        "Trainee",
        model,
        batch_size=batch_size,
        gamma=gamma,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=1,
        tau=0.005,
        lr=1e-4,
        replay_buffer_size=1024,
        use_prioritized_replay=False,
        use_double_dqn=use_double_dqn,
        q_value_clip=q_value_clip,
    )
    trainee.set_target_update_frequency(0)
    return trainee


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    """Approximate 95% CI for binomial proportion using Wilson score interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center, center - margin, center + margin


def main():
    parser = argparse.ArgumentParser(
        description="Extended validation: DQN vs GapMaximizer to test for variance"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", "-c",
        type=Path,
        help="Path to checkpoint .pt file (e.g. model_best.pt)",
    )
    group.add_argument(
        "--run", "-r",
        type=Path,
        help="Path to experiment run directory (uses models/model_best.pt or best_round.json)",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=5000,
        help="Episodes per position (total = 2 * this). Default 5000 (= 10000 total)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.run:
        run_path = args.run.resolve()
        if not run_path.is_dir():
            sys.exit(f"Run path is not a directory: {run_path}")
        checkpoint_path = None
        # Check run/models and run/workspace/models (experiment manager may use workspace)
        for cand in [run_path / "models", run_path / "workspace" / "models"]:
            if not cand.exists():
                continue
            best_pt = cand / "model_best.pt"
            if best_pt.exists():
                checkpoint_path = best_pt
                print(f"Using best model: {checkpoint_path}")
                break
            best_json = cand / "best_round.json"
            if best_json.exists():
                with open(best_json) as f:
                    best = json.load(f)
                r = best.get("best_round", 0)
                round_pt = cand / f"model_round_{r}.pt"
                if round_pt.exists():
                    checkpoint_path = round_pt
                    print(f"Using round {r} model: {checkpoint_path}")
                    break
                sys.exit(f"best_round.json found but model_round_{r}.pt missing in {cand}")
        if checkpoint_path is None:
            sys.exit(f"No models/ or workspace/models/ with model_best.pt or best_round.json in {run_path}")
    else:
        checkpoint_path = args.checkpoint.resolve()
        if not checkpoint_path.exists():
            sys.exit(f"Checkpoint not found: {checkpoint_path}")

    trainee = load_dqn_agent(checkpoint_path)
    gapmax = Players.ScoreGapMaximizer("GapMaximizer")

    print(f"\n{'='*60}")
    print("Extended Validation: DQN vs GapMaximizer")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes per position: {args.episodes} (total: {2 * args.episodes})")
    print(f"{'='*60}\n")

    tw, ow, tw_p1, tw_p2 = Training.validate_both_positions(
        trainee,
        gapmax,
        args.episodes,
        model_id_prefix=None,
        round_number=None,
        log_actions=False,
        log_metrics=False,
    )
    total = 2 * args.episodes  # trainee_first + trainee_second
    draws = total - tw - ow
    wr = tw / total if total > 0 else 0.0

    center, lo, hi = wilson_ci(tw, total)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Total games: {total}")
    print(f"Trainee wins: {tw} ({wr:.2%})")
    print(f"GapMaximizer wins: {ow} ({ow/total:.2%})" if total > 0 else "GapMaximizer wins: 0")
    if draws > 0:
        print(f"Draws: {draws} ({draws/total:.2%})")
    print(f"  As P1: {tw_p1} wins")
    print(f"  As P2: {tw_p2} wins")
    print(f"\n95%% CI (Wilson): [{lo:.2%}, {hi:.2%}]")
    print(f"{'='*60}\n")

    results = {
        "checkpoint": str(checkpoint_path),
        "episodes_per_position": args.episodes,
        "total_games": total,
        "trainee_wins": tw,
        "opponent_wins": ow,
        "draws": draws,
        "win_rate": wr,
        "win_rate_ci_95": [lo, hi],
        "wins_as_p1": tw_p1,
        "wins_as_p2": tw_p2,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
