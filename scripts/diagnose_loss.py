#!/usr/bin/env python3
"""
Diagnostic script to analyze loss behavior and identify root causes.

This script helps identify why loss is consistently rising by:
1. Checking Q-value distributions
2. Analyzing reward distributions
3. Checking target network updates
4. Verifying Bellman equation calculations
"""

import sys
import json
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import numpy as np
from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork

def analyze_q_values(model, env, num_samples=100):
    """Analyze Q-value distributions to check for anomalies."""
    print("\n" + "="*60)
    print("Q-VALUE ANALYSIS")
    print("="*60)
    
    q_values_list = []
    for _ in range(num_samples):
        env.reset()
        obs = env.get_obs()
        with torch.no_grad():
            q_vals = model(obs)
            q_values_list.append(q_vals.cpu().numpy())
    
    all_q_values = np.concatenate(q_values_list)
    
    print(f"Q-value statistics (from {num_samples} random states):")
    print(f"  Mean: {np.mean(all_q_values):.4f}")
    print(f"  Std:  {np.std(all_q_values):.4f}")
    print(f"  Min:  {np.min(all_q_values):.4f}")
    print(f"  Max:  {np.max(all_q_values):.4f}")
    print(f"  Values > 15: {np.sum(all_q_values > 15)} ({100*np.sum(all_q_values > 15)/len(all_q_values):.2f}%)")
    print(f"  Values < -15: {np.sum(all_q_values < -15)} ({100*np.sum(all_q_values < -15)/len(all_q_values):.2f}%)")
    
    return all_q_values

def check_bellman_equation(agent, gamma=0.90, q_clip=15.0):
    """Check if Bellman equation is being computed correctly."""
    print("\n" + "="*60)
    print("BELLMAN EQUATION CHECK")
    print("="*60)
    
    if len(agent.memory) < agent.batch_size:
        print(f"Insufficient samples in memory: {len(agent.memory)} < {agent.batch_size}")
        return
    
    # Sample a batch
    transitions = agent.memory.sample(agent.batch_size)
    from cuttle.players import Transition
    batch = Transition(*zip(*transitions))
    
    # Check reward distribution
    rewards = torch.cat(batch.reward).numpy()
    print(f"\nReward statistics (batch of {len(rewards)}):")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Std:  {np.std(rewards):.4f}")
    print(f"  Min:  {np.min(rewards):.4f}")
    print(f"  Max:  {np.max(rewards):.4f}")
    print(f"  Terminal rewards (1.0, -1.0, -0.5): {np.sum(np.isin(rewards, [1.0, -1.0, -0.5]))}")
    print(f"  Intermediate rewards: {np.sum(~np.isin(rewards, [1.0, -1.0, -0.5]))}")
    
    # Check next state values
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        dtype=torch.bool
    )
    non_final_count = non_final_mask.sum().item()
    terminal_count = (~non_final_mask).sum().item()
    
    print(f"\nState type distribution:")
    print(f"  Terminal states: {terminal_count}")
    print(f"  Non-terminal states: {non_final_count}")
    
    # Compute expected values for a few samples
    if non_final_count > 0:
        non_final_next_states = [s for s in batch.next_state if s is not None]
        with torch.no_grad():
            next_q_values = agent.target(non_final_next_states[:min(10, len(non_final_next_states))]).max(1).values
            next_q_values = torch.clamp(next_q_values, -q_clip, q_clip)
        
        print(f"\nNext state Q-values (sample of {len(next_q_values)}):")
        print(f"  Mean: {next_q_values.mean().item():.4f}")
        print(f"  Std:  {next_q_values.std().item():.4f}")
        print(f"  Min:  {next_q_values.min().item():.4f}")
        print(f"  Max:  {next_q_values.max().item():.4f}")

def check_target_network_sync(agent):
    """Check if target network is properly synced with policy network."""
    print("\n" + "="*60)
    print("TARGET NETWORK SYNC CHECK")
    print("="*60)
    
    # Get a sample state
    env = CuttleEnvironment()
    env.reset()
    obs = env.get_obs()
    
    with torch.no_grad():
        policy_q = agent.policy(obs)
        target_q = agent.target(obs)
    
    diff = (policy_q - target_q).abs()
    
    print(f"Policy vs Target Q-value difference:")
    print(f"  Mean absolute difference: {diff.mean().item():.4f}")
    print(f"  Max absolute difference: {diff.max().item():.4f}")
    print(f"  Target update counter: {agent.update_target_counter}")
    print(f"  Target update frequency: {agent.target_update_frequency}")
    
    if agent.target_update_frequency > 0:
        steps_since_update = agent.update_target_counter % agent.target_update_frequency
        print(f"  Steps since last update: {steps_since_update}/{agent.target_update_frequency}")

def main():
    """Run all diagnostics."""
    print("="*60)
    print("LOSS DIAGNOSTIC TOOL")
    print("="*60)
    
    # Load config
    config_file = Path(__file__).parent.parent / "hyperparams_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\nLoaded config from {config_file}")
    else:
        print("\nUsing default config")
        config = {
            "embedding_size": 16,
            "batch_size": 64,
            "gamma": 0.90,
            "learning_rate": 3e-5,
            "target_update_frequency": 500
        }
    
    # Create environment and model
    env = CuttleEnvironment(
        include_highest_point_value=False,
        include_highest_point_value_opponent_field=False
    )
    actions = env.actions
    
    model = NeuralNetwork(env.observation_space, config["embedding_size"], actions, None)
    agent = Players.Agent(
        "TestAgent", model, config["batch_size"],
        config["gamma"], 0.9, 0.05, 10000, 0.005, config["learning_rate"]
    )
    agent.set_target_update_frequency(config.get("target_update_frequency", 500))
    
    # Try to load a checkpoint if available
    checkpoint_dir = Path(__file__).parent.parent / "models"
    checkpoint_files = list(checkpoint_dir.glob("no_features_checkpoint*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.target.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print("Warning: Checkpoint format not recognized")
    else:
        print("\nNo checkpoint found, using random initialization")
    
    # Run diagnostics
    analyze_q_values(model, env)
    check_target_network_sync(agent)
    
    if len(agent.memory) >= agent.batch_size:
        check_bellman_equation(agent, config["gamma"])
    else:
        print(f"\nMemory too small for Bellman check: {len(agent.memory)} < {agent.batch_size}")
        print("Run some training episodes first to populate memory")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

