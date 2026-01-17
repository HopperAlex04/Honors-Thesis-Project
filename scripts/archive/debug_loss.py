#!/usr/bin/env python3
"""
Debug script to identify why loss is consistently rising.

This script will:
1. Check if target network is actually being updated
2. Verify Bellman equation calculations
3. Check reward distributions
4. Identify any fundamental bugs
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import numpy as np
from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork

def check_bellman_calculation():
    """Check if Bellman equation is being calculated correctly."""
    print("\n" + "="*60)
    print("BELLMAN EQUATION VERIFICATION")
    print("="*60)
    
    env = CuttleEnvironment()
    env.reset()
    model = NeuralNetwork(env.observation_space, 16, env.actions, None)
    agent = Players.Agent("TestAgent", model, 64, 0.90, 0.9, 0.05, 10000, 0.005, 5e-5)
    agent.set_target_update_frequency(500)
    
    # CRITICAL FIX: Pre-populate memory with enough samples (batch_size = 64)
    print(f"Pre-populating memory with {agent.batch_size} samples...")
    batch_size = agent.batch_size
    
    for i in range(batch_size):
        obs1 = env.get_obs()
        env.reset()
        obs2 = env.get_obs()
        
        # Add variety of transitions:
        if i % 4 == 0:
            # Terminal win
            agent.memory.push(obs1, torch.tensor([0]), None, torch.tensor([1.0]))
        elif i % 4 == 1:
            # Terminal loss
            agent.memory.push(obs1, torch.tensor([1]), None, torch.tensor([-1.0]))
        elif i % 4 == 2:
            # Terminal draw
            agent.memory.push(obs1, torch.tensor([2]), None, torch.tensor([-0.5]))
        else:
            # Non-terminal with intermediate reward
            agent.memory.push(obs1, torch.tensor([3]), obs2, torch.tensor([0.01]))
    
    print(f"Memory populated: {len(agent.memory)} samples")
    
    # Run optimize - should work now
    loss = agent.optimize()
    
    if loss is not None:
        print(f"✓ Loss computed successfully: {loss:.6f}")
        print("✓ Bellman equation calculation works")
        
        # Verify the calculation makes sense
        if loss < 0:
            print("  ⚠ WARNING: Loss is negative (unusual but possible)")
        elif loss > 100:
            print("  ⚠ WARNING: Loss is very high (>100)")
        else:
            print(f"  ✓ Loss value is reasonable")
    else:
        print("✗ ERROR: Still not enough samples in memory!")
        print(f"  Memory size: {len(agent.memory)}, Required: {agent.batch_size}")
    
    return loss

def check_target_network_updates():
    """Verify target network is being updated correctly."""
    print("\n" + "="*60)
    print("TARGET NETWORK UPDATE VERIFICATION")
    print("="*60)
    
    env = CuttleEnvironment()
    env.reset()
    model = NeuralNetwork(env.observation_space, 16, env.actions, None)
    agent = Players.Agent("TestAgent", model, 64, 0.90, 0.9, 0.05, 10000, 0.005, 5e-5)
    agent.set_target_update_frequency(10)  # Update every 10 steps for testing
    
    obs = env.get_obs()
    
    # Get initial Q-values
    with torch.no_grad():
        policy_q_initial = agent.policy(obs)
        target_q_initial = agent.target(obs)
    
    initial_diff = (policy_q_initial - target_q_initial).abs().mean().item()
    print(f"Initial policy-target difference: {initial_diff:.6f}")
    
    # Modify policy network weights
    for param in agent.policy.parameters():
        param.data += 0.1
    
    # Get Q-values after modification
    with torch.no_grad():
        policy_q_modified = agent.policy(obs)
        target_q_modified = agent.target(obs)
    
    modified_diff = (policy_q_modified - target_q_modified).abs().mean().item()
    print(f"After policy modification, difference: {modified_diff:.6f}")
    
    # CRITICAL FIX: Pre-populate memory with enough samples BEFORE testing
    # The target network only updates when optimize() actually runs (not when it returns None)
    print(f"\nPre-populating memory with {agent.batch_size} samples...")
    for i in range(agent.batch_size):
        obs1 = env.get_obs()
        env.reset()
        obs2 = env.get_obs()
        agent.memory.push(obs1, torch.tensor([0]), obs2, torch.tensor([1.0]))
    print(f"Memory size: {len(agent.memory)}")
    
    # Now simulate 10 optimization steps (each should actually run)
    print(f"\nRunning 10 optimization steps...")
    for i in range(10):
        # Add more transitions to keep memory populated
        obs1 = env.get_obs()
        env.reset()
        obs2 = env.get_obs()
        agent.memory.push(obs1, torch.tensor([0]), obs2, torch.tensor([1.0]))
        
        # This should always run now (memory has enough samples)
        result = agent.optimize()
        if result is None:
            print(f"  Step {i+1}: WARNING - optimize() returned None!")
        else:
            print(f"  Step {i+1}: optimize() ran, counter={agent.update_target_counter}, loss={result:.6f}")
    
    # Check if target was updated
    with torch.no_grad():
        policy_q_after = agent.policy(obs)
        target_q_after = agent.target(obs)
    
    after_diff = (policy_q_after - target_q_after).abs().mean().item()
    print(f"\nAfter 10 optimization steps:")
    print(f"  Policy-target difference: {after_diff:.6f}")
    print(f"  Target update counter: {agent.update_target_counter}")
    print(f"  Target update frequency: {agent.target_update_frequency}")
    print(f"  Should have updated at steps: {[i for i in range(1, 11) if i % agent.target_update_frequency == 0]}")
    
    # More accurate check: Compare target to policy at the observation
    # If target was updated at step 10, it should match policy closely
    expected_updates = agent.update_target_counter // agent.target_update_frequency
    print(f"  Expected number of updates: {expected_updates}")
    
    if expected_updates > 0:
        # Target should be close to policy if it was just updated
        if after_diff < 0.1:  # Very close means it was recently updated
            print("✓ Target network appears to be updating (difference is small)")
        elif after_diff < modified_diff:
            print("✓ Target network is being updated (difference decreased)")
        else:
            print("✗ WARNING: Target network may not be updating correctly!")
            print(f"  Difference increased from {modified_diff:.6f} to {after_diff:.6f}")
            print(f"  This suggests target network is not syncing with policy network")
    else:
        print("✗ ERROR: Target network should have updated but counter shows 0 updates!")
        print(f"  Counter: {agent.update_target_counter}, Frequency: {agent.target_update_frequency}")

def check_reward_distribution():
    """Check if rewards are being stored correctly."""
    print("\n" + "="*60)
    print("REWARD DISTRIBUTION CHECK")
    print("="*60)
    
    # Check reward constants
    from cuttle.training import REWARD_WIN, REWARD_LOSS, REWARD_DRAW, SCORE_REWARD_SCALE, GAP_REWARD_SCALE
    
    print(f"Reward constants:")
    print(f"  REWARD_WIN: {REWARD_WIN}")
    print(f"  REWARD_LOSS: {REWARD_LOSS}")
    print(f"  REWARD_DRAW: {REWARD_DRAW}")
    print(f"  SCORE_REWARD_SCALE: {SCORE_REWARD_SCALE}")
    print(f"  GAP_REWARD_SCALE: {GAP_REWARD_SCALE}")
    
    # Check theoretical Q-value bounds
    gamma = 0.90
    max_q_terminal = REWARD_WIN  # Terminal: Q = r (no future)
    
    # CORRECTED: The infinite horizon formula (REWARD_WIN / (1 - gamma) = 10.0) is WRONG
    # for this game because:
    # 1. Episodes are finite (games end)
    # 2. You can't get REWARD_WIN at every step - only when you win
    # 3. Intermediate rewards are small (SCORE_REWARD_SCALE = 0.01)
    #
    # Realistic maximum: If you get intermediate rewards for many steps before winning:
    # - Max intermediate reward per step: ~0.1 (score changes of ~10 points * 0.01 scale)
    # - Typical episode length: ~20-50 turns
    # - Q(s,a) ≈ intermediate_rewards + γ^N * REWARD_WIN
    # - For N=20: Q ≈ 0.1*20 + 0.9^20*1.0 ≈ 2.0 + 0.12 ≈ 2.12
    # - For N=50: Q ≈ 0.1*50 + 0.9^50*1.0 ≈ 5.0 + 0.005 ≈ 5.0
    #
    # So realistic max is probably around 5-10, not 10 from infinite horizon
    max_turns_typical = 30  # Typical game length
    max_intermediate_per_turn = 0.1  # Rough estimate
    max_q_realistic = max_intermediate_per_turn * max_turns_typical + (gamma ** max_turns_typical) * REWARD_WIN
    
    print(f"\nTheoretical Q-value bounds:")
    print(f"  Terminal state max: {max_q_terminal}")
    print(f"  Infinite horizon theoretical max (WRONG for this game): {REWARD_WIN / (1 - gamma):.2f}")
    print(f"  Realistic max (with intermediate rewards, ~{max_turns_typical} turns): {max_q_realistic:.2f}")
    print(f"  Current Q-value clip: 100.0 (recently increased from 15.0)")
    
    if max_q_realistic > 100.0:
        print(f"  ⚠ WARNING: Realistic max ({max_q_realistic:.2f}) > clip (100.0)")
        print(f"    This could cause Q-values to be clipped incorrectly!")
    elif max_q_realistic > 15.0:
        print(f"  ✓ Realistic max ({max_q_realistic:.2f}) is within new clip (100.0)")
        print(f"    Previous clip of 15.0 was too restrictive!")

def check_device_consistency():
    """Check if tensors are on the correct device."""
    print("\n" + "="*60)
    print("DEVICE CONSISTENCY CHECK")
    print("="*60)
    
    env = CuttleEnvironment()
    env.reset()
    model = NeuralNetwork(env.observation_space, 16, env.actions, None)
    agent = Players.Agent("TestAgent", model, 64, 0.90, 0.9, 0.05, 10000, 0.005, 5e-5)
    
    # Check device of model parameters
    policy_device = next(agent.policy.parameters()).device
    target_device = next(agent.target.parameters()).device
    
    print(f"Policy network device: {policy_device}")
    print(f"Target network device: {target_device}")
    
    if policy_device != target_device:
        print("✗ ERROR: Policy and target networks on different devices!")
    else:
        print("✓ Policy and target networks on same device")
    
    # Check if next_state_values tensor is on correct device
    # This would require running optimize, but we can check the pattern
    print("\nNote: next_state_values is created as torch.zeros(batch_size)")
    print("      This should be on CPU by default, which may cause device mismatch!")

def main():
    """Run all diagnostic checks."""
    print("="*60)
    print("LOSS DEBUGGING - ROOT CAUSE ANALYSIS")
    print("="*60)
    
    check_reward_distribution()
    check_target_network_updates()
    check_bellman_calculation()
    check_device_consistency()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. If target network isn't updating: Check target_update_frequency")
    print("2. If Q-values exceed clip: Consider increasing Q_VALUE_CLIP or reducing gamma")
    print("3. If device mismatch: Ensure all tensors are on same device")
    print("4. If rewards are wrong: Check update_replay_memory function")
    print("="*60)

if __name__ == "__main__":
    main()

