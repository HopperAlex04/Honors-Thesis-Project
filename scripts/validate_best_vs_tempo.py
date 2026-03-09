#!/usr/bin/env python3
"""
Run best PPO and DQN models for 500 games in each seat (P1 and P2) against TempoPlayer.

Usage:
    python scripts/validate_best_vs_tempo.py
    python scripts/validate_best_vs_tempo.py --episodes 1000
    python scripts/validate_best_vs_tempo.py --ppo-run path/to/ppo_run --dqn-run path/to/dqn_run --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from cuttle import players as Players
from cuttle import training as Training
from cuttle.environment import CuttleEnvironment
from cuttle.networks import EmbeddingActorCritic, EmbeddingBasedNeuralNetwork, NeuralNetwork


# Default best runs (by validation vs GapMaximizer)
DEFAULT_PPO_RUN = project_root / "experiments/experiment_20260219_084225_ppo/runs/ppo_large_hidden_run_01"
DEFAULT_DQN_RUN = project_root / "experiments/experiment_20260222_142807_best_storage_match/runs/large_hidden_embedding_run_02"


def build_dqn_from_config(cfg, observation_space, num_actions, use_position_indicator: bool = True):
    """Build DQN model from config (matches train/validate_dqn)."""
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


def resolve_dqn_checkpoint(run_path: Path) -> Path:
    """Resolve checkpoint from run dir: models/model_best.pt or best_round.json -> model_round_N.pt."""
    for cand in [run_path / "models", run_path / "workspace" / "models"]:
        if not cand.exists():
            continue
        best_pt = cand / "model_best.pt"
        if best_pt.exists():
            return best_pt
        best_json = cand / "best_round.json"
        if best_json.exists():
            with open(best_json) as f:
                best = json.load(f)
            r = best.get("best_round", 0)
            round_pt = cand / f"model_round_{r}.pt"
            if round_pt.exists():
                return round_pt
            raise FileNotFoundError(f"best_round.json says round {r} but {round_pt} missing")
    raise FileNotFoundError(f"No models/ or workspace/models/ with model_best.pt or best_round.json in {run_path}")


def load_dqn_agent(checkpoint_path: Path) -> Players.Agent:
    """Load DQN agent from checkpoint (greedy)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    use_pos = True
    state = ckpt.get("model_state_dict", {})
    for k, v in state.items():
        if ("hidden_layers.0.weight" in k or "output_layer.weight" in k) and v.shape[1] == 468:
            use_pos = False
            break
    cfg = {**cfg, "use_position_indicator": use_pos}
    env = CuttleEnvironment()
    model = build_dqn_from_config(cfg, env.observation_space, env.actions)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    trainee = Players.Agent(
        "Trainee",
        model,
        batch_size=cfg.get("batch_size", 128),
        gamma=cfg.get("gamma", 0.9),
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=1,
        tau=0.005,
        lr=1e-4,
        replay_buffer_size=1024,
        use_prioritized_replay=False,
        use_double_dqn=cfg.get("use_double_dqn", False),
        q_value_clip=cfg.get("q_value_clip", 15.0),
    )
    trainee.set_target_update_frequency(0)
    return trainee


def load_ppo_agent(run_path: Path, checkpoint_name: str = "model_best.pt") -> Players.PPOAgent:
    """Load PPO agent from run dir (models/model_best.pt + hyperparams_config.json)."""
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
    ppo_config = config.get("ppo", {})
    ppo_hidden = ppo_config.get("hidden_layers", [128, 128])
    embedding_dim = config.get("embedding_dim", 52)
    zone_encoded_dim = config.get("zone_encoded_dim", 52)
    use_position_indicator = ppo_config.get("use_position_indicator", False)
    model = EmbeddingActorCritic(
        env.observation_space,
        num_actions=env.actions,
        embedding_dim=embedding_dim,
        zone_encoded_dim=zone_encoded_dim,
        hidden_layers=ppo_hidden,
        use_position_indicator=use_position_indicator,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    return trainee


def main():
    parser = argparse.ArgumentParser(
        description="Run best PPO and DQN vs TempoPlayer (500 games per seat each)"
    )
    parser.add_argument(
        "--ppo-run",
        type=Path,
        default=DEFAULT_PPO_RUN,
        help=f"PPO run directory (default: {DEFAULT_PPO_RUN.name})",
    )
    parser.add_argument(
        "--dqn-run",
        type=Path,
        default=DEFAULT_DQN_RUN,
        help=f"DQN run directory (default: {DEFAULT_DQN_RUN.name})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Episodes per position (total per model = 2 * this). Default 500",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save combined results to JSON",
    )
    args = parser.parse_args()

    ppo_run = args.ppo_run.resolve()
    dqn_run = args.dqn_run.resolve()

    opponent = Players.TempoPlayer("Tempo")
    episodes = args.episodes
    total_games = 2 * episodes

    all_results = {}

    # --- PPO vs Tempo ---
    print("\n" + "=" * 60)
    print("Loading best PPO model...")
    print("=" * 60)
    if not ppo_run.is_dir():
        print(f"PPO run not found: {ppo_run}", file=sys.stderr)
    else:
        ppo_trainee = load_ppo_agent(ppo_run)
        print(f"PPO run: {ppo_run.name}")
        print(f"Running {episodes} games per seat vs Tempo (total {total_games})...")
        tw, ow, tw_p1, tw_p2 = Training.validate_both_positions(
            ppo_trainee,
            opponent,
            episodes,
            model_id_prefix=None,
            round_number=None,
            log_actions=False,
            log_metrics=False,
        )
        wr = tw / total_games if total_games > 0 else 0.0
        print(f"\nPPO vs Tempo: Trainee wins {tw}/{total_games} ({wr:.2%})  (P1: {tw_p1}, P2: {tw_p2})")
        all_results["ppo"] = {
            "run": str(ppo_run),
            "episodes_per_position": episodes,
            "total_games": total_games,
            "trainee_wins": tw,
            "opponent_wins": ow,
            "win_rate": wr,
            "wins_as_p1": tw_p1,
            "wins_as_p2": tw_p2,
        }

    # --- DQN vs Tempo ---
    print("\n" + "=" * 60)
    print("Loading best DQN model...")
    print("=" * 60)
    if not dqn_run.is_dir():
        print(f"DQN run not found: {dqn_run}", file=sys.stderr)
    else:
        dqn_ckpt = resolve_dqn_checkpoint(dqn_run)
        dqn_trainee = load_dqn_agent(dqn_ckpt)
        print(f"DQN run: {dqn_run.name} (checkpoint: {dqn_ckpt.name})")
        print(f"Running {episodes} games per seat vs Tempo (total {total_games})...")
        tw, ow, tw_p1, tw_p2 = Training.validate_both_positions(
            dqn_trainee,
            opponent,
            episodes,
            model_id_prefix=None,
            round_number=None,
            log_actions=False,
            log_metrics=False,
        )
        wr = tw / total_games if total_games > 0 else 0.0
        print(f"\nDQN vs Tempo: Trainee wins {tw}/{total_games} ({wr:.2%})  (P1: {tw_p1}, P2: {tw_p2})")
        all_results["dqn"] = {
            "run": str(dqn_run),
            "checkpoint": str(dqn_ckpt),
            "episodes_per_position": episodes,
            "total_games": total_games,
            "trainee_wins": tw,
            "opponent_wins": ow,
            "win_rate": wr,
            "wins_as_p1": tw_p1,
            "wins_as_p2": tw_p2,
        }

    print("\n" + "=" * 60)
    print("Summary: Best PPO and DQN vs TempoPlayer")
    print("=" * 60)
    for name, r in all_results.items():
        print(f"  {name.upper()}: {r['trainee_wins']}/{r['total_games']} ({r['win_rate']:.2%})")
    print("=" * 60 + "\n")

    if args.output and all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
