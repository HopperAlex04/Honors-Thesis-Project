#!/usr/bin/env python3
"""
Interactive script for a human to play against a trained model.

Usage:
    python play_against_model.py
    python play_against_model.py --checkpoint models/no_features_checkpoint0.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# Limit CPU threads (must be set before importing torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import torch
torch.set_num_threads(4)

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork
from cuttle import training as Training


def card_to_string(card_index: int) -> str:
    """Convert card index to human-readable string (e.g., 'Ace of Spades')."""
    suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
    ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
    
    suit = card_index // 13
    rank = card_index % 13
    
    return f"{ranks[rank]} of {suits[suit]}"


def display_game_state(env: CuttleEnvironment, human_is_p1: bool = True):
    """Display the current game state in a readable format."""
    obs = env.get_obs()
    # Get scores from both perspectives
    # scoreState() returns (score, threshold)
    p1_score, p1_threshold = env.scoreState()  # Current perspective (P1)
    env.passControl()
    p2_score, p2_threshold = env.scoreState()  # Other perspective (P2)
    env.passControl()  # Switch back to original perspective
    
    print("\n" + "="*60)
    print("GAME STATE")
    print("="*60)
    
    # Get zones from observation
    # Observation uses "Current Zones" dict and "Off-Player Field"
    current_zones = obs.get("Current Zones", {})
    off_player_field = obs.get("Off-Player Field", np.zeros(52, dtype=bool))
    
    # Determine which player is which
    if human_is_p1:
        human_name = "You (P1)"
        ai_name = "AI (P2)"
        human_hand = current_zones.get("Hand", np.zeros(52, dtype=bool))
        human_field = current_zones.get("Field", np.zeros(52, dtype=bool))
        ai_field = off_player_field
    else:
        human_name = "You (P2)"
        ai_name = "AI (P1)"
        # When human is P2, current_zones is P1's zones, off_player_field is P2's field
        human_hand = off_player_field  # This is approximate - P2's hand isn't directly visible
        human_field = off_player_field  # P2's field
        ai_field = current_zones.get("Field", np.zeros(52, dtype=bool))  # P1's field
    
    # Display scores
    print(f"\nScores:")
    print(f"  {human_name}: {p1_score if human_is_p1 else p2_score}")
    print(f"  {ai_name}: {p2_score if human_is_p1 else p1_score}")
    
    # Display human's hand
    hand_cards = [i for i in range(52) if human_hand[i]]
    print(f"\n{human_name} Hand ({len(hand_cards)} cards):")
    if hand_cards:
        for i, card in enumerate(hand_cards):
            print(f"  [{i}] {card_to_string(card)}")
    else:
        print("  (empty)")
    
    # Display human's field
    field_cards = [i for i in range(52) if human_field[i]]
    print(f"\n{human_name} Field ({len(field_cards)} cards):")
    if field_cards:
        for card in field_cards:
            print(f"  - {card_to_string(card)}")
    else:
        print("  (empty)")
    
    # Display AI's field
    ai_field_cards = [i for i in range(52) if ai_field[i]]
    print(f"\n{ai_name} Field ({len(ai_field_cards)} cards):")
    if ai_field_cards:
        for card in ai_field_cards:
            print(f"  - {card_to_string(card)}")
    else:
        print("  (empty)")
    
    # Display stack (now boolean array - check if any cards are in stack)
    stack = obs.get("Stack", np.zeros(52, dtype=bool))
    if isinstance(stack, np.ndarray) and np.any(stack):
        stack_cards = [i for i in range(52) if stack[i]]
        print(f"\nStack: {len(stack_cards)} card(s) - {', '.join(card_to_string(c) for c in stack_cards)}")
    
    print("="*60 + "\n")


def action_to_string(env: CuttleEnvironment, action_id: int) -> str:
    """Convert action ID to human-readable string."""
    action_obj = env.action_registry.get_action(action_id)
    if action_obj is None:
        return f"Unknown action {action_id}"
    
    action_type = action_obj.__class__.__name__
    args = action_obj.args
    
    if action_type == "DrawAction":
        return "Draw a card"
    elif action_type == "ScoreAction":
        card = args if isinstance(args, int) else args[0] if isinstance(args, (list, tuple)) else None
        if card is not None:
            return f"Score {card_to_string(card)}"
        return f"Score (card {args})"
    elif action_type == "ScuttleAction":
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            card = args[0]
            target = args[1]
            return f"Scuttle {card_to_string(target)} with {card_to_string(card)}"
        return f"Scuttle (action {action_id})"
    elif action_type == "AceAction":
        card = args if isinstance(args, int) else args[0] if isinstance(args, (list, tuple)) else None
        if card is not None:
            return f"Play Ace {card_to_string(card)}"
        return f"Play Ace (card {args})"
    elif action_type == "TwoAction":
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            card = args[0]
            target = args[1]
            return f"Play Two {card_to_string(card)} on {card_to_string(target)}"
        return f"Play Two (action {action_id})"
    elif action_type == "ThreeAction":
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            card = args[0]
            target = args[1]
            return f"Play Three {card_to_string(card)} on {card_to_string(target)}"
        return f"Play Three (action {action_id})"
    elif action_type == "FourAction":
        card = args if isinstance(args, int) else args[0] if isinstance(args, (list, tuple)) else None
        if card is not None:
            return f"Play Four {card_to_string(card)}"
        return f"Play Four (card {args})"
    elif action_type == "FiveAction":
        card = args if isinstance(args, int) else args[0] if isinstance(args, (list, tuple)) else None
        if card is not None:
            return f"Play Five {card_to_string(card)}"
        return f"Play Five (card {args})"
    elif action_type == "SixAction":
        card = args if isinstance(args, int) else args[0] if isinstance(args, (list, tuple)) else None
        if card is not None:
            return f"Play Six {card_to_string(card)}"
        return f"Play Six (card {args})"
    elif action_type == "SevenAction":
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            card = args[0]
            target = args[1]
            return f"Play Seven {card_to_string(card)} on {card_to_string(target)}"
        return f"Play Seven (action {action_id})"
    elif action_type == "NineAction":
        if isinstance(args, (list, tuple)) and len(args) >= 3:
            card = args[0]
            target = args[1]
            self_hit = args[2]
            target_str = "yourself" if self_hit else card_to_string(target)
            return f"Play Nine {card_to_string(card)} on {target_str}"
        return f"Play Nine (action {action_id})"
    else:
        return f"{action_type} (action {action_id})"


def get_human_action(env: CuttleEnvironment, valid_actions: List[int]) -> int:
    """Get action from human player via command line input."""
    print("\nValid actions:")
    action_descriptions = []
    for i, action_id in enumerate(valid_actions):
        desc = action_to_string(env, action_id)
        action_descriptions.append((action_id, desc))
        print(f"  [{i}] {desc} (ID: {action_id})")
    
    while True:
        try:
            choice = input("\nSelect action (enter number): ").strip()
            if choice == "":
                print("Please enter a number.")
                continue
            
            index = int(choice)
            if 0 <= index < len(valid_actions):
                selected_action = valid_actions[index]
                print(f"\nYou selected: {action_to_string(env, selected_action)}")
                return selected_action
            else:
                print(f"Please enter a number between 0 and {len(valid_actions) - 1}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nGame interrupted by user.")
            sys.exit(0)


def load_model(checkpoint_path: Path, config_path: Optional[Path] = None) -> Tuple[Players.Agent, CuttleEnvironment]:
    """Load a trained model from checkpoint."""
    # Load config if available
    if config_path is None:
        config_path = Path(__file__).parent / "hyperparams_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "embedding_size": 16,
            "batch_size": 128,
            "gamma": 0.90,
            "eps_start": 0.90,
            "eps_end": 0.05,
            "eps_decay": 28510,
            "tau": 0.005,
            "target_update_frequency": 500,
            "learning_rate": 3e-5,
            "replay_buffer_size": 30000,
        }
    
    # Create environment (no feature flags needed anymore)
    env = CuttleEnvironment()
    actions = env.actions
    
    # Create model
    embedding_size = config.get("embedding_size", 16)
    model = NeuralNetwork(env.observation_space, embedding_size, actions, None)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    else:
        # Old format - checkpoint is the model itself
        model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully (old format)")
    
    # Create agent
    batch_size = config.get("batch_size", 128)
    gamma = config.get("gamma", 0.90)
    eps_start = config.get("eps_start", 0.90)
    eps_end = config.get("eps_end", 0.05)
    eps_decay = config.get("eps_decay", 28510)
    tau = config.get("tau", 0.005)
    lr = config.get("learning_rate", 3e-5)
    replay_buffer_size = config.get("replay_buffer_size", 30000)
    
    agent = Players.Agent(
        "AI", model, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, replay_buffer_size
    )
    
    # Set agent to greedy mode (no exploration)
    agent.eps_end = 0.0
    agent.eps_start = 0.0
    
    return agent, env


def list_checkpoints(models_dir: Path) -> List[Path]:
    """List all available checkpoint files."""
    checkpoints = []
    for file in models_dir.glob("*.pt"):
        if file.is_file():
            checkpoints.append(file)
    return sorted(checkpoints)


def select_checkpoint(models_dir: Path) -> Path:
    """Interactive checkpoint selection."""
    checkpoints = list_checkpoints(models_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}")
        sys.exit(1)
    
    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        size = checkpoint.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  [{i}] {checkpoint.name} ({size:.2f} MB)")
    
    while True:
        try:
            choice = input(f"\nSelect checkpoint (0-{len(checkpoints)-1}): ").strip()
            index = int(choice)
            if 0 <= index < len(checkpoints):
                return checkpoints[index]
            else:
                print(f"Please enter a number between 0 and {len(checkpoints) - 1}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nExiting.")
            sys.exit(0)


class HumanPlayer(Players.Player):
    """Human player that gets actions via command line input."""
    
    def __init__(self, name: str, env: CuttleEnvironment):
        super().__init__(name)
        self.env = env
    
    def getAction(
        self,
        observation: dict,
        valid_actions: List[int],
        total_actions: int,
        steps_done: int,
        force_greedy: bool = False
    ) -> int:
        """Get action from human via command line."""
        return get_human_action(self.env, valid_actions)


def play_game(agent: Players.Agent, env: CuttleEnvironment, human_is_p1: bool = True):
    """Main game loop using training helper functions."""
    print("\n" + "="*60)
    print("GAME START")
    print("="*60)
    print(f"You are playing as {'Player 1' if human_is_p1 else 'Player 2'}")
    print("The AI will play as the other player.")
    print("="*60 + "\n")
    
    # Create player objects
    human_player = HumanPlayer("Human", env)
    ai_player = agent
    
    # Set up players based on position
    if human_is_p1:
        p1 = human_player
        p2 = ai_player
    else:
        p1 = ai_player
        p2 = human_player
    
    # Reset environment
    env.reset()
    # After reset, environment is in P1's perspective
    turn = 0
    episode = 0
    
    # Game loop - following the pattern from training code
    while True:
        turn += 1
        
        # Get scores from both perspectives
        # scoreState() returns (score, threshold), not (current, opponent)!
        # We need to get scores from both perspectives by switching
        current_score, current_threshold = env.scoreState()
        env.passControl()
        opponent_score, opponent_threshold = env.scoreState()
        env.passControl()  # Switch back to original perspective
        
        # Determine which perspective we're in to map scores correctly
        is_p1_perspective = (turn % 2 == 1) or (turn == 1)
        
        if is_p1_perspective:
            p1_score, p2_score = current_score, opponent_score
            p1_threshold, p2_threshold = current_threshold, opponent_threshold
        else:
            p1_score, p2_score = opponent_score, current_score
            p1_threshold, p2_threshold = opponent_threshold, current_threshold
        
        # Check if game is over (using thresholds)
        if p1_score >= p1_threshold or p2_score >= p2_threshold:
            display_game_state(env, human_is_p1)
            if p1_score >= p1_threshold:
                if human_is_p1:
                    print("🎉 You win!")
                else:
                    print("🤖 AI wins!")
            elif p2_score >= p2_threshold:
                if human_is_p1:
                    print("🤖 AI wins!")
                else:
                    print("🎉 You win!")
            else:
                print("Game ended in a draw.")
            break
        
        display_game_state(env, human_is_p1)
        
        # Determine current player (alternating turns)
        # Player 1 goes first (odd turns), Player 2 goes second (even turns)
        # After reset, we're in P1's perspective
        # After P1's turn, we passControl to P2's perspective
        is_p1_turn = (turn % 2 == 1)
        
        # Ensure we're in the correct perspective
        # For P1's turn (odd), we should be in P1's perspective
        # For P2's turn (even), we should be in P2's perspective (after passControl from previous turn)
        if is_p1_turn:
            # Make sure we're in P1's perspective
            # If we're not (shouldn't happen after reset, but be safe), switch back
            # Actually, after reset and for odd turns, we should already be in P1's perspective
            pass  # We're already in the right perspective
        else:
            # For P2's turn, we should be in P2's perspective (from previous turn's passControl)
            # But if this is turn 2 and we haven't passed control yet, do it now
            if turn == 2:
                # First time switching to P2, need to pass control
                env.passControl()
        
        # Determine which player object corresponds to current turn
        current_player = p1 if is_p1_turn else p2
        other_player = p2 if is_p1_turn else p1
        current_is_human = (current_player == human_player)
        
        print(f"\n--- Turn {turn}: {'Your' if current_is_human else 'AI'}'s Turn ---")
        
        # Execute turn using training helper
        try:
            player_states = []
            player_actions = []
            other_states = []
            other_actions = []
            
            observation, score, terminated, truncated = Training.execute_player_turn(
                env=env,
                player=current_player,
                other_player=other_player,
                actions=env.actions,
                steps=0,
                validating=True,  # Use greedy policy
                player_states=player_states,
                player_actions=player_actions,
                other_states=other_states,
                other_actions=other_actions,
                episode=episode,
                turn=turn,
                action_logger=None
            )
            
            if terminated or truncated:
                display_game_state(env, human_is_p1)
                # Get scores from both perspectives to determine winner
                # scoreState() returns (score, threshold)
                current_score, current_threshold = env.scoreState()
                env.passControl()
                opponent_score, opponent_threshold = env.scoreState()
                env.passControl()  # Switch back
                
                if is_p1_turn:
                    p1_score, p2_score = current_score, opponent_score
                    p1_threshold, p2_threshold = current_threshold, opponent_threshold
                else:
                    p1_score, p2_score = opponent_score, current_score
                    p1_threshold, p2_threshold = opponent_threshold, current_threshold
                
                if p1_score >= p1_threshold:
                    if human_is_p1:
                        print("🎉 You win!")
                    else:
                        print("🤖 AI wins!")
                elif p2_score >= p2_threshold:
                    if human_is_p1:
                        print("🤖 AI wins!")
                    else:
                        print("🎉 You win!")
                else:
                    print("Game ended in a draw.")
                break
            
            # After turn, prepare for next player's turn
            # Following training code pattern (from selfPlayTraining):
            #   env.emptyStack()
            #   env.passControl()
            #   env.end_turn()
            # This switches to the other player's perspective for the next turn
            env.emptyStack()
            env.passControl()  # Switch to other player's perspective
            env.end_turn()  # End the current turn
            
        except KeyboardInterrupt:
            print("\n\nGame interrupted by user.")
            raise
        except Exception as e:
            print(f"Error during turn: {e}")
            import traceback
            traceback.print_exc()
            # Try to continue
            continue
        
        # Small delay for readability
        import time
        time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="Play against a trained Cuttle model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints (default: models)"
    )
    parser.add_argument(
        "--human-p1",
        action="store_true",
        default=True,
        help="Human plays as Player 1 (default: True)"
    )
    parser.add_argument(
        "--human-p2",
        action="store_true",
        help="Human plays as Player 2"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)
    
    # Select checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = select_checkpoint(models_dir)
    
    # Determine human player position
    human_is_p1 = not args.human_p2 if args.human_p2 else args.human_p1
    
    # Load model
    try:
        agent, env = load_model(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Play game
    try:
        play_game(agent, env, human_is_p1)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during gameplay: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

