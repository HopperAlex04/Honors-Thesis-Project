#!/usr/bin/env python3
"""
Run a few PPO training episodes and verify optimize() returns correct metrics dict.
Use: from project root, python tests/test_ppo_metrics.py
"""
import json
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import EmbeddingActorCritic
from cuttle import training as Training


def main():
    print("Loading config...")
    config_path = project_root / "hyperparams_config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["algorithm"] = "ppo"
    ppo_config = config.get("ppo", {})
    embedding_dim = config.get("embedding_dim", 52)
    zone_encoded_dim = config.get("zone_encoded_dim", 52)
    ppo_hidden = ppo_config.get("hidden_layers", [128, 128])

    print("Building env and PPO model...")
    env = CuttleEnvironment()
    actions = env.actions
    model = EmbeddingActorCritic(
        env.observation_space,
        num_actions=actions,
        embedding_dim=embedding_dim,
        zone_encoded_dim=zone_encoded_dim,
        hidden_layers=ppo_hidden,
        use_position_indicator=ppo_config.get("use_position_indicator", False),
    )
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
    opponent = Players.ScoreGapMaximizer("GapMaximizer")

    num_episodes = 25
    optimize_every = 10
    print(f"Running {num_episodes} episodes (optimize every {optimize_every})...")
    # Run training; metrics go to metrics_logger. We'll run with log_metrics=True
    # and a known model_id, then read the jsonl.
    import tempfile
    import os
    orig_metrics_dir = Training.METRICS_LOG_DIRECTORY
    test_dir = Path(tempfile.mkdtemp(prefix="ppo_metrics_test_"))
    try:
        Training.METRICS_LOG_DIRECTORY = test_dir
        p1_wins, p2_wins, steps = Training.selfPlayTraining(
            trainee,
            opponent,
            num_episodes,
            validating=False,
            log_actions=False,
            log_metrics=True,
            model_id="test_ppo_metrics",
            initial_steps=0,
            round_number=0,
            initial_total_time=0.0,
            early_stopping_config={"enabled": False},
            reward_mode="binary",
            optimize_every_n_episodes=optimize_every,
        )
        # File is log_dir / "metrics_<model_id>.jsonl"
        path = test_dir / "metrics_test_ppo_metrics.jsonl"
        assert path.exists(), f"Metrics file not found: {path}"
        print(f"\nReading metrics from {path}")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == num_episodes, f"Expected {num_episodes} lines, got {len(lines)}"

        expected_ppo_keys = {
            "loss", "ppo_policy_loss", "ppo_value_loss", "ppo_entropy",
            "ppo_n_steps", "ppo_n_chunks", "ppo_advantage_mean", "ppo_advantage_std",
            "ppo_ratio_mean", "ppo_ratio_min", "ppo_ratio_max", "ppo_ratio_std",
            "ppo_clip_fraction",
        }
        episodes_with_optimize = 0
        for i, line in enumerate(lines):
            rec = json.loads(line)
            assert "episode" in rec and "loss" in rec
            if rec.get("loss") is not None:
                episodes_with_optimize += 1
                # When loss is present, PPO metrics should be present
                for k in expected_ppo_keys:
                    assert k in rec, f"Episode {i} (optimize): missing key {k}"
                loss = rec["loss"]
                assert isinstance(loss, (int, float)), f"loss should be scalar, got {type(loss)}"
                assert rec["ppo_n_steps"] > 0, "ppo_n_steps should be > 0"
                assert rec["ppo_n_chunks"] > 0, "ppo_n_chunks should be > 0"
                assert 0 <= rec["ppo_clip_fraction"] <= 1, "ppo_clip_fraction should be in [0,1]"
                print(f"  Episode {i}: loss={loss:.4f} n_steps={rec['ppo_n_steps']} n_chunks={rec['ppo_n_chunks']} "
                      f"ratio mean={rec['ppo_ratio_mean']:.3f} min={rec['ppo_ratio_min']:.3f} max={rec['ppo_ratio_max']:.3f} "
                      f"clip_frac={rec['ppo_clip_fraction']:.3f}")

        assert episodes_with_optimize >= 2, f"Expected at least 2 optimize episodes, got {episodes_with_optimize}"
        print(f"\nOK: {episodes_with_optimize} episodes had PPO updates, all metrics present and sane.")
    finally:
        Training.METRICS_LOG_DIRECTORY = orig_metrics_dir
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
