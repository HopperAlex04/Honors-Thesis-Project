#!/usr/bin/env python3
"""
Analyze training efficiency and determine optimal epsilon decay rate.

This script analyzes:
1. Current epsilon decay vs performance
2. When agents reach good performance levels
3. Optimal decay rates for faster training turnaround
4. Recommendations for shortening training

Usage:
    python scripts/analyze_training_efficiency.py
    python scripts/analyze_training_efficiency.py --type hand_only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VALID_TRAINING_TYPES = [
    "hand_only",
    "opponent_field_only",
    "no_features",
    "both_features",  # Legacy name, kept for backward compatibility
    "all_features",
    "scores"
]


def calculate_epsilon(steps: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    """Calculate epsilon value at given step count."""
    return eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_decay)


def load_performance_data(
    base_dir: Path,
    training_type: str,
    opponent_type: str
) -> Dict[int, Dict[str, float]]:
    """Load performance data for analysis."""
    log_dir = base_dir / "action_logs" / training_type
    
    if not log_dir.exists():
        return {}
    
    metrics_by_round = {}
    
    first_pattern = f"metrics_{training_type}_round_*_{opponent_type}_trainee_first.jsonl"
    second_pattern = f"metrics_{training_type}_round_*_{opponent_type}_trainee_second.jsonl"
    
    first_files = {}
    for f in log_dir.glob(first_pattern):
        import re
        match = re.search(r'_round_(\d+)_', f.name)
        if match:
            first_files[int(match.group(1))] = f
    
    second_files = {}
    for f in log_dir.glob(second_pattern):
        import re
        match = re.search(r'_round_(\d+)_', f.name)
        if match:
            second_files[int(match.group(1))] = f
    
    for round_num in sorted(set(first_files.keys()) | set(second_files.keys())):
        first_last = None
        second_last = None
        
        if round_num in first_files:
            with open(first_files[round_num]) as f:
                for line in f:
                    if line.strip():
                        try:
                            first_last = json.loads(line)
                        except:
                            pass
        
        if round_num in second_files:
            with open(second_files[round_num]) as f:
                for line in f:
                    if line.strip():
                        try:
                            second_last = json.loads(line)
                        except:
                            pass
        
        if first_last and second_last:
            trainee_wins = first_last.get('p1_wins', 0) + second_last.get('p2_wins', 0)
            opponent_wins = first_last.get('p2_wins', 0) + second_last.get('p1_wins', 0)
            total = trainee_wins + opponent_wins
            
            if total > 0:
                win_rate = trainee_wins / total
                metrics_by_round[round_num] = {'win_rate': win_rate}
    
    return metrics_by_round


def load_epsilon_data(base_dir: Path, training_type: str) -> Dict[int, Dict[str, float]]:
    """Load epsilon values from selfplay metrics."""
    log_dir = base_dir / "action_logs" / training_type
    epsilon_by_round = {}
    
    for round_num in range(20):  # Check up to 20 rounds
        metrics_file = log_dir / f"metrics_{training_type}_round_{round_num}_selfplay.jsonl"
        if metrics_file.exists():
            epsilons = []
            steps_list = []
            with open(metrics_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get('p1_epsilon') is not None:
                                epsilons.append(data['p1_epsilon'])
                                steps_list.append(data.get('total_steps', 0))
                        except:
                            pass
            
            if epsilons:
                epsilon_by_round[round_num] = {
                    'epsilon': epsilons[-1],
                    'steps': steps_list[-1] if steps_list else 0
                }
    
    return epsilon_by_round


def find_performance_milestones(
    performance_data: Dict[int, Dict[str, float]],
    milestones: List[float] = [0.5, 0.55, 0.6, 0.65]
) -> Dict[float, Optional[int]]:
    """Find when agent reaches performance milestones."""
    rounds = sorted(performance_data.keys())
    win_rates = [performance_data[r]['win_rate'] for r in rounds]
    
    results = {}
    for milestone in milestones:
        reached_round = None
        for r, wr in zip(rounds, win_rates):
            if wr >= milestone and reached_round is None:
                reached_round = r
                break
        results[milestone] = reached_round
    
    return results


def simulate_decay_scenarios(
    current_eps_decay: int,
    target_rounds: int,
    steps_per_round: int
) -> Dict[str, Dict]:
    """Simulate different decay scenarios."""
    EPS_START = 0.90
    EPS_END = 0.05
    
    scenarios = {}
    
    # Current scenario
    total_steps = target_rounds * steps_per_round
    final_epsilon_current = calculate_epsilon(total_steps, EPS_START, EPS_END, current_eps_decay)
    scenarios['current'] = {
        'eps_decay': current_eps_decay,
        'final_epsilon': final_epsilon_current,
        'rounds_to_10pct': None,
        'rounds_to_20pct': None
    }
    
    # Find when epsilon reaches 10% and 20% with current decay
    for r in range(1, target_rounds + 1):
        steps = r * steps_per_round
        eps = calculate_epsilon(steps, EPS_START, EPS_END, current_eps_decay)
        if eps <= 0.10 and scenarios['current']['rounds_to_10pct'] is None:
            scenarios['current']['rounds_to_10pct'] = r
        if eps <= 0.20 and scenarios['current']['rounds_to_20pct'] is None:
            scenarios['current']['rounds_to_20pct'] = r
    
    # Faster decay scenarios
    for factor in [0.5, 0.6, 0.7, 0.8]:
        new_decay = int(current_eps_decay * factor)
        final_epsilon = calculate_epsilon(total_steps, EPS_START, EPS_END, new_decay)
        
        rounds_to_10pct = None
        rounds_to_20pct = None
        for r in range(1, target_rounds + 1):
            steps = r * steps_per_round
            eps = calculate_epsilon(steps, EPS_START, EPS_END, new_decay)
            if eps <= 0.10 and rounds_to_10pct is None:
                rounds_to_10pct = r
            if eps <= 0.20 and rounds_to_20pct is None:
                rounds_to_20pct = r
        
        scenarios[f'{int(factor*100)}pct'] = {
            'eps_decay': new_decay,
            'final_epsilon': final_epsilon,
            'rounds_to_10pct': rounds_to_10pct,
            'rounds_to_20pct': rounds_to_20pct
        }
    
    return scenarios


def analyze_training_efficiency(
    base_dir: Path,
    training_type: str = "hand_only"
) -> None:
    """Perform comprehensive training efficiency analysis."""
    print(f"\n{'='*80}")
    print(f"TRAINING EFFICIENCY ANALYSIS: {training_type.upper()}")
    print(f"{'='*80}\n")
    
    # Load epsilon data
    epsilon_data = load_epsilon_data(base_dir, training_type)
    if not epsilon_data:
        print(f"No epsilon data found for {training_type}")
        return
    
    # Calculate average steps per round
    rounds = sorted(epsilon_data.keys())
    if len(rounds) < 2:
        print("Insufficient data for analysis")
        return
    
    steps_per_round = []
    for i in range(1, len(rounds)):
        steps_diff = epsilon_data[rounds[i]]['steps'] - epsilon_data[rounds[i-1]]['steps']
        steps_per_round.append(steps_diff)
    
    avg_steps_per_round = int(np.mean(steps_per_round)) if steps_per_round else 12000
    
    print(f"Average steps per round: {avg_steps_per_round}")
    print(f"Total rounds analyzed: {len(rounds)}")
    print()
    
    # Current settings
    EPS_START = 0.90
    EPS_END = 0.05
    EPS_DECAY = 80000
    
    print(f"Current Settings:")
    print(f"  EPS_START: {EPS_START}")
    print(f"  EPS_END: {EPS_END}")
    print(f"  EPS_DECAY: {EPS_DECAY}")
    print()
    
    # Analyze epsilon decay
    print("Epsilon Decay Analysis:")
    print(f"{'Round':<8} {'Steps':<10} {'Epsilon':<12} {'Exploration %':<15}")
    print("-" * 50)
    for r in rounds[:10]:
        steps = epsilon_data[r]['steps']
        eps = epsilon_data[r]['epsilon']
        print(f"{r:<8} {steps:<10} {eps:<12.3f} {eps*100:<15.1f}%")
    print()
    
    # Analyze performance milestones
    print("Performance Milestones:")
    opponents = ['vs_heuristic', 'vs_randomized', 'vs_gapmaximizer']
    
    for opponent in opponents:
        perf_data = load_performance_data(base_dir, training_type, opponent)
        if perf_data:
            milestones = find_performance_milestones(perf_data)
            print(f"\n{opponent.upper()}:")
            for milestone, round_num in milestones.items():
                if round_num is not None:
                    print(f"  Reached {milestone*100:.0f}% win rate at round {round_num}")
                else:
                    print(f"  Never reached {milestone*100:.0f}% win rate")
    
    print()
    
    # Simulate different decay scenarios
    print("Decay Scenario Comparison:")
    print(f"{'Scenario':<15} {'EPS_DECAY':<12} {'Final ε':<12} {'Rounds to 20%':<15} {'Rounds to 10%':<15}")
    print("-" * 75)
    
    scenarios = simulate_decay_scenarios(EPS_DECAY, len(rounds), avg_steps_per_round)
    
    for name, data in scenarios.items():
        name_display = name.replace('pct', '%').title()
        final_eps = f"{data['final_epsilon']:.3f}"
        rounds_20 = f"{data['rounds_to_20pct']}" if data['rounds_to_20pct'] else "N/A"
        rounds_10 = f"{data['rounds_to_10pct']}" if data['rounds_to_10pct'] else "N/A"
        print(f"{name_display:<15} {data['eps_decay']:<12} {final_eps:<12} {rounds_20:<15} {rounds_10:<15}")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    # Find when performance plateaus
    best_opponent = None
    best_data = None
    for opponent in opponents:
        perf_data = load_performance_data(base_dir, training_type, opponent)
        if perf_data and len(perf_data) > 3:
            rounds = sorted(perf_data.keys())
            win_rates = [perf_data[r]['win_rate'] for r in rounds]
            
            # Check if performance plateaus (last 3 rounds similar)
            if len(win_rates) >= 3:
                recent_avg = np.mean(win_rates[-3:])
                earlier_avg = np.mean(win_rates[-6:-3]) if len(win_rates) >= 6 else win_rates[0]
                improvement = recent_avg - earlier_avg
                
                if best_data is None or abs(improvement) < abs(best_data['improvement']):
                    best_data = {
                        'opponent': opponent,
                        'rounds': rounds,
                        'win_rates': win_rates,
                        'improvement': improvement,
                        'recent_avg': recent_avg
                    }
    
    if best_data:
        print(f"\n1. Performance Analysis ({best_data['opponent']}):")
        print(f"   - Recent average win rate: {best_data['recent_avg']:.3f}")
        print(f"   - Improvement in last 3 rounds: {best_data['improvement']:+.3f}")
        
        if abs(best_data['improvement']) < 0.02:
            print(f"   → Performance has plateaued - training could stop earlier")
        
        # Find optimal stopping point
        rounds = best_data['rounds']
        win_rates = best_data['win_rates']
        
        # Find round where performance stabilizes (within 2% of final)
        final_rate = win_rates[-1]
        stable_round = None
        for i, (r, wr) in enumerate(zip(rounds, win_rates)):
            if abs(wr - final_rate) <= 0.02 and stable_round is None:
                stable_round = r
                print(f"   → Performance stabilizes at round {stable_round} (within 2% of final)")
                break
    
    # Epsilon recommendations
    print(f"\n2. Epsilon Decay Recommendations:")
    
    final_round = rounds[-1] if rounds else 9
    final_steps = epsilon_data[final_round]['steps'] if final_round in epsilon_data else final_round * avg_steps_per_round
    final_epsilon = epsilon_data[final_round]['epsilon'] if final_round in epsilon_data else calculate_epsilon(final_steps, EPS_START, EPS_END, EPS_DECAY)
    
    print(f"   - Current: ε = {final_epsilon:.3f} ({final_epsilon*100:.1f}% exploration) at round {final_round}")
    print(f"   - This is still quite high - agent is exploring {final_epsilon*100:.1f}% of the time")
    
    # Calculate faster decay options
    target_epsilon_early = 0.10  # 10% exploration
    target_epsilon_late = 0.20   # 20% exploration
    
    # Find decay rate that reaches 10% by round 5
    target_round = 5
    target_steps = target_round * avg_steps_per_round
    
    # Solve: 0.10 = 0.05 + (0.90 - 0.05) * exp(-target_steps / eps_decay)
    # 0.05 = 0.85 * exp(-target_steps / eps_decay)
    # eps_decay = -target_steps / ln(0.05/0.85)
    faster_decay_10pct = int(-target_steps / np.log(0.05 / 0.85))
    
    # For 20% by round 5
    faster_decay_20pct = int(-target_steps / np.log(0.15 / 0.85))
    
    print(f"\n3. Faster Decay Options:")
    print(f"   Option A: EPS_DECAY = {faster_decay_20pct} (reaches 20% exploration by round {target_round})")
    print(f"            → {((EPS_DECAY - faster_decay_20pct) / EPS_DECAY * 100):.0f}% faster decay")
    print(f"            → Final ε at round {final_round}: {calculate_epsilon(final_steps, EPS_START, EPS_END, faster_decay_20pct):.3f}")
    
    print(f"\n   Option B: EPS_DECAY = {faster_decay_10pct} (reaches 10% exploration by round {target_round})")
    print(f"            → {((EPS_DECAY - faster_decay_10pct) / EPS_DECAY * 100):.0f}% faster decay")
    print(f"            → Final ε at round {final_round}: {calculate_epsilon(final_steps, EPS_START, EPS_END, faster_decay_10pct):.3f}")
    
    print(f"\n4. Training Length Recommendations:")
    
    # Check when good performance is reached
    for opponent in ['vs_heuristic', 'vs_randomized']:
        perf_data = load_performance_data(base_dir, training_type, opponent)
        if perf_data:
            milestones = find_performance_milestones(perf_data, [0.5, 0.6])
            reached_60 = milestones.get(0.6)
            if reached_60 is not None:
                print(f"   - {opponent}: Reaches 60% win rate by round {reached_60}")
                print(f"     → Could potentially stop training after round {reached_60 + 2} (2 rounds for stability)")
    
    print(f"\n   - Current training: {final_round + 1} rounds")
    if best_data and stable_round:
        potential_savings = final_round - stable_round
        if potential_savings > 0:
            print(f"   - Performance stabilizes at round {stable_round}")
            print(f"   - Potential savings: {potential_savings} rounds ({potential_savings/(final_round+1)*100:.0f}% reduction)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training efficiency and recommend optimal decay rates"
    )
    parser.add_argument(
        "--type", "-t",
        choices=VALID_TRAINING_TYPES,
        default="hand_only",
        help="Training type to analyze"
    )
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        default=Path("."),
        help="Base directory containing action_logs"
    )
    
    args = parser.parse_args()
    
    analyze_training_efficiency(args.base_dir, args.type)
    return 0


if __name__ == "__main__":
    sys.exit(main())

