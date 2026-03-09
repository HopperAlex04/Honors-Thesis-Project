#!/usr/bin/env python3
"""
Run GapMaximizer vs GapMaximizer for 1000 games to measure the inherent positional
balance of Cuttle (P1 vs P2 advantage from the rules alone).

Usage:
    python scripts/run_gapmaximizer_balance.py [--games 1000] [--output results.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root so we can import cuttle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from cuttle import players as Players
from cuttle import training as Training


def main():
    parser = argparse.ArgumentParser(description="GapMaximizer vs GapMaximizer to measure positional balance")
    parser.add_argument("--games", "-n", type=int, default=1000, help="Number of games to run (default: 1000)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    p1 = Players.ScoreGapMaximizer("GapMaximizer")
    p2 = Players.ScoreGapMaximizer("GapMaximizer")

    print(f"Running GapMaximizer vs GapMaximizer for {args.games} games...")
    p1_wins, p2_wins, _ = Training.selfPlayTraining(
        p1, p2, args.games,
        validating=True,
        log_actions=False,
        log_metrics=False,
    )
    draws = args.games - p1_wins - p2_wins

    p1_wr = p1_wins / args.games
    p2_wr = p2_wins / args.games
    draw_rate = draws / args.games

    print(f"\n{'='*60}")
    print("GapMaximizer vs GapMaximizer — Positional Balance")
    print(f"{'='*60}")
    print(f"Games: {args.games}")
    print(f"P1 (first player, 5 cards) wins: {p1_wins} ({p1_wr:.1%})")
    print(f"P2 (second player/dealer, 6 cards) wins: {p2_wins} ({p2_wr:.1%})")
    print(f"Draws: {draws} ({draw_rate:.1%})")
    print(f"{'='*60}")
    if p1_wr > p2_wr:
        print(f"P1 has +{(p1_wr - p2_wr)*100:.1f} pp advantage")
    elif p2_wr > p1_wr:
        print(f"P2 has +{(p2_wr - p1_wr)*100:.1f} pp advantage")
    else:
        print("Positions are balanced")

    results = {
        "games": args.games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_wr,
        "p2_win_rate": p2_wr,
        "draw_rate": draw_rate,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
