#!/usr/bin/env python3
"""
Analyze the loss spike at episode 1000.

This script helps identify what's happening around episode 1000 that causes
the massive loss spike.
"""

import sys
import json
import math
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def analyze_episode_1000_events():
    """Analyze what events happen around episode 1000."""
    print("="*60)
    print("EPISODE 1000 EVENT ANALYSIS")
    print("="*60)
    
    # Load config
    config_file = Path(__file__).parent.parent / "hyperparams_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "eps_decay": 28510,
            "eps_start": 0.90,
            "eps_end": 0.05,
            "learning_rate": 3e-5,
            "lr_decay_rate": 0.9,
            "lr_decay_interval": 2,
            "replay_buffer_size": 25000,
            "target_update_frequency": 500,
            "training": {
                "rounds": 10,
                "eps_per_round": 500
            }
        }
    
    eps_per_round = config["training"].get("eps_per_round", 500)
    rounds = config["training"].get("rounds", 10)
    buffer_size = config.get("replay_buffer_size", 25000)
    lr_decay_interval = config.get("lr_decay_interval", 2)
    
    # Calculate round boundaries
    print(f"\nRound Structure:")
    for r in range(rounds):
        start_ep = r * eps_per_round
        end_ep = (r + 1) * eps_per_round - 1
        print(f"  Round {r}: Episodes {start_ep}-{end_ep}")
        if start_ep <= 1000 <= end_ep:
            print(f"    ⚠ Episode 1000 is in Round {r}")
    
    # Check LR decay schedule
    print(f"\nLearning Rate Decay Schedule:")
    print(f"  Decay interval: Every {lr_decay_interval} rounds")
    lr_decay_rounds = [r for r in range(0, rounds) if r > 0 and r % lr_decay_interval == 0]
    print(f"  LR decays at rounds: {lr_decay_rounds}")
    for r in lr_decay_rounds:
        start_ep = r * eps_per_round
        print(f"    Round {r} (episode {start_ep}): LR decays by {config.get('lr_decay_rate', 0.9)}")
        if start_ep == 1000:
            print(f"      ⚠ THIS IS THE SPIKE! LR decays at exactly episode 1000")
    
    # Check replay buffer saturation
    print(f"\nReplay Buffer Analysis:")
    print(f"  Buffer size: {buffer_size:,}")
    experiences_per_episode = 40  # Estimate
    episodes_to_fill = buffer_size / experiences_per_episode
    print(f"  Estimated experiences per episode: {experiences_per_episode}")
    print(f"  Episodes to fill buffer: {episodes_to_fill:.0f}")
    if episodes_to_fill < 1000:
        episodes_since_full = 1000 - episodes_to_fill
        print(f"  At episode 1000: Buffer has been full for {episodes_since_full:.0f} episodes")
        print(f"    ⚠ Old experiences from episodes 0-{episodes_to_fill:.0f} are being pushed out")
        print(f"    ⚠ This creates a sudden shift in data distribution")
    
    # Check epsilon decay
    print(f"\nEpsilon Decay Analysis:")
    eps_start = config.get("eps_start", 0.90)
    eps_end = config.get("eps_end", 0.05)
    eps_decay = config.get("eps_decay", 28510)
    
    # Estimate steps at episode 1000
    steps_per_episode = 30  # Estimate
    steps_at_1000 = 1000 * steps_per_episode
    
    epsilon_at_1000 = eps_end + (eps_start - eps_end) * math.exp(-steps_at_1000 / eps_decay)
    print(f"  Steps per episode (estimate): {steps_per_episode}")
    print(f"  Steps at episode 1000: ~{steps_at_1000:,}")
    print(f"  Epsilon at episode 1000: {epsilon_at_1000:.3f} ({epsilon_at_1000*100:.1f}% exploration)")
    
    # Check target network updates
    print(f"\nTarget Network Update Schedule:")
    target_freq = config.get("target_update_frequency", 500)
    updates_by_1000 = steps_at_1000 // target_freq
    print(f"  Update frequency: Every {target_freq} steps")
    print(f"  Updates by episode 1000: ~{updates_by_1000}")
    print(f"  Next update after episode 1000: At step {((updates_by_1000 + 1) * target_freq)}")
    
    # Summary
    print(f"\n" + "="*60)
    print("ROOT CAUSE OF SPIKE AT EPISODE 1000:")
    print("="*60)
    print("1. LEARNING RATE DECAY:")
    print(f"   - LR decays at Round 2 (episode {2 * eps_per_round})")
    print(f"   - Network suddenly learns 10% slower")
    print(f"   - But data distribution is still changing rapidly")
    print(f"   - Creates mismatch: slower learning + faster distribution shift")
    print()
    print("2. REPLAY BUFFER SATURATION:")
    print(f"   - Buffer filled at episode ~{episodes_to_fill:.0f}")
    print(f"   - By episode 1000, old experiences are being pushed out")
    print(f"   - Sudden shift from 'old easy experiences' to 'new hard experiences'")
    print(f"   - Network trained on old data, now sees completely different data")
    print()
    print("3. COMBINED EFFECT:")
    print("   - LR decay + buffer shift = Perfect storm")
    print("   - Network can't adapt fast enough to new data distribution")
    print("   - Loss spikes as network struggles to fit new experiences")
    print()
    print("SOLUTIONS:")
    print("="*60)
    print("1. Delay LR decay: Change lr_decay_interval to 3 or 4")
    print("2. Smooth buffer transition: Use larger buffer or clear gradually")
    print("3. Monitor win rates: If win rates improve, spike is acceptable")
    print("4. Reduce buffer size further: 15,000-20,000 for faster aging")

if __name__ == "__main__":
    analyze_episode_1000_events()

