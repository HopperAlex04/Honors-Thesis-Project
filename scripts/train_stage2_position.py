#!/usr/bin/env python3
"""
Stage 2 position-specific training: load a model from a completed experiment and
continue training vs GapMaximizer, with 75% on the focus side and 25% on the other
to reduce catastrophic forgetting.

Usage:
    python scripts/train_stage2_position.py \\
        --source experiments/experiment_20260219_084225_ppo/runs/ppo_large_hidden_run_03 \\
        --side p2 \\
        [--checkpoint model_best.pt] \\
        [--rounds 10] [--eps-per-round 500] \\
        [--focus-ratio 0.75] \\
        [--output experiments/exp/runs/run_01/stage2_p2]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root (before other imports that use it)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from cuttle import players as Players
from cuttle import training as Training
from cuttle.environment import CuttleEnvironment
from cuttle.networks import EmbeddingActorCritic


def load_ppo_from_checkpoint(
    run_path: Path,
    checkpoint_name: str = "model_best.pt",
) -> tuple:
    """Load PPO model and agent from an experiment run. Returns (trainee, config, env)."""
    models_dir = run_path / "models"
    checkpoint_path = models_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = run_path / "hyperparams_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    env = CuttleEnvironment()
    actions = env.actions

    # Build PPO model from config
    ppo_config = config.get("ppo", {})
    ppo_hidden = ppo_config.get("hidden_layers", [128, 128])
    embedding_dim = config.get("embedding_dim", 52)
    zone_encoded_dim = config.get("zone_encoded_dim", 52)
    use_position_indicator = ppo_config.get("use_position_indicator", False)

    model = EmbeddingActorCritic(
        env.observation_space,
        num_actions=actions,
        embedding_dim=embedding_dim,
        zone_encoded_dim=zone_encoded_dim,
        hidden_layers=ppo_hidden,
        use_position_indicator=use_position_indicator,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")

    trainee = Players.PPOAgent(
        "PlayerAgent",
        model,
        lr=ppo_config.get("learning_rate", 3e-4),
        gamma=ppo_config.get("gamma", 0.99),
        clip_eps=ppo_config.get("clip_eps", 0.2),
        ppo_epochs=ppo_config.get("ppo_epochs", 4),
        value_coef=ppo_config.get("value_coef", 0.5),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
    )

    if "optimizer_state_dict" in checkpoint and hasattr(trainee, "optimizer"):
        trainee.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state")

    return trainee, config, env


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: train loaded model vs GapMaximizer on a specific side (P1 or P2)"
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        required=True,
        help="Path to completed experiment run (e.g. experiments/exp/runs/ppo_large_hidden_run_01)",
    )
    parser.add_argument(
        "--side",
        choices=["p1", "p2"],
        required=True,
        help="Which position to train on: p1 (first player) or p2 (second player/dealer)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="model_best.pt",
        help="Checkpoint file name (default: model_best.pt). Also: model_final.pt, model_round_N.pt",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=10,
        help="Number of training rounds (default: 10)",
    )
    parser.add_argument(
        "--eps-per-round", "-e",
        type=int,
        default=1000,
        help="Episodes per round (default: 500)",
    )
    parser.add_argument(
        "--focus-ratio",
        type=float,
        default=0.75,
        help="Fraction of episodes on focus side, rest on other side to reduce forgetting (default: 0.75)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for stage 2 model (default: <source>/stage2_<side>)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation after each round",
    )
    args = parser.parse_args()

    run_path = args.source.resolve()
    if not run_path.is_dir():
        sys.exit(f"Source is not a directory: {run_path}")

    output_dir = args.output or (run_path / f"stage2_{args.side}")
    output_dir = output_dir.resolve()
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_logs").mkdir(parents=True, exist_ok=True)

    # Change to output dir so training logs (metrics_logs, action_logs) land there
    orig_cwd = os.getcwd()
    os.chdir(output_dir)

    trainee, config, env = load_ppo_from_checkpoint(run_path, args.checkpoint)

    optimize_every = config.get("ppo", {}).get("optimize_every_n_episodes", 10)
    reward_mode = config.get("training", {}).get("reward_mode", "binary")
    gapmax = Players.ScoreGapMaximizer("GapMaximizer")

    focus_side = args.side.lower()
    focus_eps = int(args.eps_per_round * args.focus_ratio)
    other_eps = args.eps_per_round - focus_eps

    print(f"\n{'='*60}")
    print(f"Stage 2: Training {args.focus_ratio:.0%} {focus_side.upper()} / {1 - args.focus_ratio:.0%} other")
    print(f"  Source: {run_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rounds: {args.rounds}, Episodes/round: {args.eps_per_round} ({focus_eps} focus + {other_eps} other)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    steps_done = 0
    total_time = 0.0
    win_rate_history = {"GapMaximizer": []}

    for round_num in range(args.rounds):
        round_start = time.time()
        print(f"\nRound {round_num + 1}/{args.rounds}")

        # Focus side: e.g. P2 = trainee as second
        p1_focus = gapmax if focus_side == "p2" else trainee
        p2_focus = trainee if focus_side == "p2" else gapmax
        _, _, steps_done = Training.selfPlayTraining(
            p1_focus, p2_focus, focus_eps,
            model_id=f"stage2_{args.side}_round_{round_num}_focus",
            initial_steps=steps_done,
            round_number=round_num,
            initial_total_time=total_time,
            validating=False,
            log_actions=False,
            log_metrics=not args.no_validation,
            early_stopping_config={"enabled": False},
            reward_mode=reward_mode,
            optimize_every_n_episodes=optimize_every,
        )

        # Other side: 25% to reduce forgetting
        p1_other = trainee if focus_side == "p2" else gapmax
        p2_other = gapmax if focus_side == "p2" else trainee
        _, _, steps_done = Training.selfPlayTraining(
            p1_other, p2_other, other_eps,
            model_id=f"stage2_{args.side}_round_{round_num}_other",
            initial_steps=steps_done,
            round_number=round_num,
            initial_total_time=total_time,
            validating=False,
            log_actions=False,
            log_metrics=not args.no_validation,
            early_stopping_config={"enabled": False},
            reward_mode=reward_mode,
            optimize_every_n_episodes=optimize_every,
        )

        total_time += time.time() - round_start

        if not args.no_validation:
            trainee_wins, _, wins_p1, wins_p2 = Training.validate_both_positions(
                trainee, gapmax, 125,
                model_id_prefix=f"stage2_{args.side}_round_{round_num}_val",
                round_number=round_num,
                initial_total_time=total_time,
            )
            wr = trainee_wins / 250.0
            win_rate_history["GapMaximizer"].append(wr)
            print(f"  Validation: {trainee_wins}/250 ({wr:.1%}) [P1: {wins_p1/125:.1%}, P2: {wins_p2/125:.1%}]")

    # Final validation: 1000 episodes per side
    if not args.no_validation:
        print(f"\nFinal validation (1000 episodes per side)...")
        tw, _, wp1, wp2 = Training.validate_both_positions(
            trainee, gapmax, 1000,
            model_id_prefix=f"stage2_{args.side}_final_val",
            round_number=args.rounds,
            initial_total_time=total_time,
        )
        print(f"  Final: {tw}/2000 ({tw/2000:.1%}) [P1: {wp1/1000:.1%}, P2: {wp2/1000:.1%}]")

    # Save final model
    out_path = output_dir / "models" / "model_stage2.pt"
    torch.save({
        "model_state_dict": trainee.model.state_dict(),
        "optimizer_state_dict": trainee.get_optimizer_state(),
        "config": config,
        "stage": "stage2",
        "side": args.side,
        "focus_ratio": args.focus_ratio,
        "rounds": args.rounds,
        "eps_per_round": args.eps_per_round,
        "source_run": str(run_path),
    }, out_path)
    print(f"\nSaved stage 2 model: {out_path}")

    os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
