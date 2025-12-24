"""
Training module for self-play reinforcement learning.

This module provides functionality for training DQN agents through self-play
in the Cuttle card game environment, with comprehensive logging capabilities
for actions and training metrics.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment


# Constants
MAX_TURNS_PER_EPISODE = 200  # Safety limit to prevent infinite loops
SLOW_TURN_THRESHOLD = 0.1  # Log turns taking longer than this (in seconds)
MAX_COUNTER_DEPTH = 4  # Maximum depth for counter exchanges (stack indices 0-4)
STACK_TOP_SEVEN = 7  # Stack value indicating Seven card resolution needed
STACK_TOP_FOUR = 4  # Stack value indicating Four card resolution needed
LOG_DIRECTORY = Path("./action_logs")

# Reward constants
REWARD_WIN = 1.0  # Reward for winning an episode
REWARD_LOSS = -1.0  # Reward for losing an episode
REWARD_DRAW = -0.5  # Reward for drawing an episode (penalize heavily to discourage passive play)
REWARD_INTERMEDIATE = 0.0  # Reward for intermediate steps (non-terminal states)


def setup_logger(
    logger_name: str, 
    log_file: Path, 
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with file handler.
    
    Args:
        logger_name: Name for the logger
        log_file: Path to the log file
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.handlers = []  # Remove existing handlers to avoid duplicates
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def setup_action_logger(log_dir: Path, model_id: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Set up action logging for strategy analysis.
    
    Args:
        log_dir: Directory for log files
        model_id: Optional model identifier (e.g., "round_3", "checkpoint_5")
        
    Returns:
        Configured logger or None if logging disabled
    """
    log_dir.mkdir(exist_ok=True)
    if model_id:
        log_file = log_dir / f"actions_{model_id}.jsonl"
    else:
        log_file = log_dir / f"actions.jsonl"
    logger = setup_logger("action_logger", log_file)
    print(f"Action logging enabled: {log_file}")
    return logger


def setup_metrics_logger(log_dir: Path, model_id: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Set up metrics logging for training statistics.
    
    Args:
        log_dir: Directory for log files
        model_id: Optional model identifier (e.g., "round_3", "checkpoint_5")
        
    Returns:
        Configured logger or None if logging disabled
    """
    log_dir.mkdir(exist_ok=True)
    if model_id:
        metrics_file = log_dir / f"metrics_{model_id}.jsonl"
    else:
        metrics_file = log_dir / f"metrics.jsonl"
    logger = setup_logger("metrics_logger", metrics_file)
    print(f"Metrics logging enabled: {metrics_file}")
    return logger


def extract_state_features(state: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract key features from game state for logging.
    
    Args:
        state: Game observation dictionary
        
    Returns:
        Dictionary of extracted state features
    """
    return {
        "hand_size": int(state["Current Zones"]["Hand"].sum()),
        "field_size": int(state["Current Zones"]["Field"].sum()),
        "opponent_field_size": int(state["Off-Player Field"].sum()),
        "deck_size": int(state["Deck"].sum()),
        "scrap_size": int(state["Scrap"].sum()),
        "stack_top": state["Stack"][0] if isinstance(state["Stack"], list) else 0,
    }


def log_action(
    episode: int,
    turn: int,
    player_name: str,
    action_id: int,
    state_features_before: Dict[str, int],
    state_features_after: Dict[str, int],
    score_before: int,
    score_after: int,
    valid_actions: List[int],
    action_logger: Optional[logging.Logger],
    env: CuttleEnvironment,
    countering: bool = False,
    reward: Optional[float] = None,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    """
    Log an action for strategy analysis.
    
    Args:
        episode: Current episode number
        turn: Current turn number
        player_name: Name of the player taking the action
        action_id: ID of the action taken
        state_features_before: Pre-extracted state features BEFORE action
        state_features_after: State features AFTER action execution
        score_before: Player's score before the action
        score_after: Player's score after the action
        valid_actions: List of valid action IDs
        action_logger: Logger instance (None to skip logging)
        env: Game environment instance
        countering: Whether this is a counter action
        reward: Reward received (if any)
        terminated: Whether the game terminated
        truncated: Whether the game was truncated
    """
    if action_logger is None:
        return
    
    # Get action object for more details
    action_obj = env.action_registry.get_action(action_id)
    action_type = action_obj.__class__.__name__ if action_obj else "Unknown"
    action_args = action_obj.args if action_obj else None
    
    log_entry = {
        "episode": episode,
        "turn": turn,
        "player": player_name,
        "action_id": action_id,
        "action_type": action_type,
        "action_args": action_args,
        "valid_actions_count": len(valid_actions),
        "countering": countering,
        "state_before": state_features_before,
        "state_after": state_features_after,
        "score_before": score_before,
        "score_after": score_after,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    
    action_logger.info(json.dumps(log_entry))


def handle_counter_exchange(
    env: CuttleEnvironment,
    current_player: Players.Player,
    other_player: Players.Player,
    actions: int,
    steps: int,
    validating: bool,
    player_states: List[Dict[str, Any]],
    player_actions: List[int],
    other_states: List[Dict[str, Any]],
    other_actions: List[int],
) -> bool:
    """
    Handle counter exchange between players.
    
    Args:
        env: Game environment
        current_player: Player who initiated the action
        other_player: Opponent player
        actions: Total number of actions
        steps: Total training steps
        validating: Whether in validation mode
        player_states: States list for current player
        player_actions: Actions list for current player
        other_states: States list for other player
        other_actions: Actions list for other player
        
    Returns:
        True if zones are currently swapped (need to swap back), False otherwise
    """
    depth = 1
    # Track whether we're on original player's side (True) or opponent's side (False)
    on_original_side = True
    
    while env.checkResponses() and depth <= MAX_COUNTER_DEPTH:
        # Switch sides
        on_original_side = not on_original_side
        env.passControl()
        
        # Get counter action from appropriate player
        active_player = current_player if on_original_side else other_player
        valid_actions = env.generateActionMask(countering=True)
        observation = env.get_obs()
        response = active_player.getAction(
            observation, valid_actions, actions, steps, force_greedy=validating
        )
        env.step(response)
        
        # Record action to appropriate list
        if on_original_side:
            player_states.append(observation)
            player_actions.append(response)
        else:
            other_states.append(observation)
            other_actions.append(response)
        
        env.updateStack(response, depth)
        depth += 1
    
    # Return True if zones are swapped (not on original side)
    return not on_original_side


def handle_stack_resolution(
    env: CuttleEnvironment,
    current_player: Players.Player,
    other_player: Players.Player,
    actions: int,
    steps: int,
    validating: bool,
    current_states: List[Dict[str, Any]],
    current_actions: List[int],
    other_states: List[Dict[str, Any]],
    other_actions: List[int],
    episode: int,
    turn: int,
    action_logger: Optional[logging.Logger],
) -> Tuple[Dict[str, Any], int, bool, bool]:
    """
    Handle stack resolution for special cards (Seven, Four).
    
    Args:
        env: Game environment
        current_player: Player whose turn it is
        other_player: Opponent player
        actions: Total number of actions
        steps: Total training steps
        validating: Whether in validation mode
        current_states: States list for current player
        current_actions: Actions list for current player
        other_states: States list for other player
        other_actions: Actions list for other player
        episode: Current episode number
        turn: Current turn number
        action_logger: Logger for action logging
        
    Returns:
        Tuple of (observation, score, terminated, truncated)
    """
    stack_top = env.stackTop()
    observation = None
    score = 0
    terminated = False
    truncated = False
    
    if stack_top == STACK_TOP_SEVEN:
        # Seven is resolved by current player
        valid_actions = env.generateActionMask()
        observation = env.get_obs()
        current_states.append(observation)
        # Capture state features BEFORE action (avoids aliasing)
        state_features_before = extract_state_features(observation)
        score_before, _ = env.scoreState()
        action = current_player.getAction(observation, valid_actions, actions, steps, force_greedy=False)
        current_actions.append(action)
        observation, score, terminated, truncated = env.step(action)
        # Capture state features AFTER action
        state_features_after = extract_state_features(observation)
        
        if action_logger:
            log_action(
                episode=episode,
                turn=turn,
                player_name=current_player.name,
                action_id=action,
                state_features_before=state_features_before,
                state_features_after=state_features_after,
                score_before=score_before,
                score_after=score,
                valid_actions=valid_actions,
                action_logger=action_logger,
                env=env,
                terminated=terminated,
                truncated=truncated,
            )
    
    elif stack_top == STACK_TOP_FOUR:
        # Four is resolved by other player (after passControl)
        env.passControl()
        observation = env.get_obs()
        other_states.append(observation)
        valid_actions = env.generateActionMask()
        # Capture state features BEFORE action (avoids aliasing)
        state_features_before = extract_state_features(observation)
        score_before, _ = env.scoreState()
        action = other_player.getAction(observation, valid_actions, actions, steps, force_greedy=validating)
        other_actions.append(action)
        observation, score, terminated, truncated = env.step(action)
        # Capture state features AFTER action (before passControl back)
        state_features_after = extract_state_features(observation)
        env.passControl()
        
        if action_logger:
            log_action(
                episode=episode,
                turn=turn,
                player_name=other_player.name,
                action_id=action,
                state_features_before=state_features_before,
                state_features_after=state_features_after,
                score_before=score_before,
                score_after=score,
                valid_actions=valid_actions,
                action_logger=action_logger,
                env=env,
                terminated=terminated,
                truncated=truncated,
            )
    
    return observation, score, terminated, truncated


def update_replay_memory(
    player: Players.Player,
    states: List[Dict[str, Any]],
    actions: List[int],
    reward: float,
    next_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update player's replay memory with episode transitions.
    
    Args:
        player: Player whose memory to update
        states: List of states from the episode
        actions: List of actions taken
        reward: Reward for the episode
        next_state: Next state (if available)
    """
    for i in range(len(states)):
        action_tensor = torch.tensor([actions[i]])
        reward_tensor = torch.tensor([reward])
        next_state_tensor = next_state if next_state is not None else None
        player.memory.push(states[i], action_tensor, next_state_tensor, reward_tensor)


def log_episode_outcome(
    episode: int,
    turn: int,
    outcome: str,
    p1_score: Optional[int],
    p2_score: Optional[int],
    reason: Optional[str],
    action_logger: Optional[logging.Logger],
) -> None:
    """
    Log episode outcome to action logger.
    
    Args:
        episode: Episode number
        turn: Final turn number
        outcome: Outcome string ("draw", "p1_win", "p2_win")
        p1_score: Player 1's final score
        p2_score: Player 2's final score
        reason: Reason for outcome (if applicable)
        action_logger: Logger instance
    """
    if action_logger is None:
        return
    
    outcome_entry = {
        "episode": episode,
        "outcome": outcome,
        "final_turn": turn,
        "p1_final_score": p1_score,
        "p2_final_score": p2_score,
    }
    if reason:
        outcome_entry["reason"] = reason
    
    action_logger.info(json.dumps(outcome_entry))


def log_episode_metrics(
    episode: int,
    p1: Players.Player,
    p2: Players.Player,
    p1_wins: int,
    p2_wins: int,
    draws: int,
    steps: int,
    turns: int,
    p1_score: int,
    p2_score: int,
    validating: bool,
    loss: Optional[float],
    metrics_logger: Optional[logging.Logger],
) -> None:
    """
    Log episode-level training metrics.
    
    Args:
        episode: Episode number
        p1: Player 1 instance
        p2: Player 2 instance
        p1_wins: Number of wins for player 1
        p2_wins: Number of wins for player 2
        draws: Number of draws
        steps: Total training steps
        turns: Turns in this episode
        p1_score: Player 1's final score
        p2_score: Player 2's final score
        validating: Whether in validation mode
        loss: Training loss (if available)
        metrics_logger: Logger instance
    """
    if metrics_logger is None:
        return
    
    total_episodes = episode + 1
    episode_stats = {
        "episode": episode,
        "p1_name": p1.name,
        "p2_name": p2.name,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_wins / total_episodes,
        "p2_win_rate": p2_wins / total_episodes,
        "draw_rate": draws / total_episodes,
        "total_steps": steps,
        "episode_turns": turns,
        "validating": validating,
        "loss": loss,
    }
    
    # Add agent-specific metrics
    if isinstance(p1, Players.Agent):
        episode_stats["p1_epsilon"] = p1.eps_end + (p1.eps_start - p1.eps_end) * math.exp(
            -1.0 * steps / p1.eps_decay
        )
        episode_stats["p1_memory_size"] = len(p1.memory)
    
    if isinstance(p2, Players.Agent):
        episode_stats["p2_epsilon"] = p2.eps_end + (p2.eps_start - p2.eps_end) * math.exp(
            -1.0 * steps / p2.eps_decay
        )
        episode_stats["p2_memory_size"] = len(p2.memory)
    
    metrics_logger.info(json.dumps(episode_stats))


def execute_player_turn(
    env: CuttleEnvironment,
    player: Players.Player,
    other_player: Players.Player,
    actions: int,
    steps: int,
    validating: bool,
    player_states: List[Dict[str, Any]],
    player_actions: List[int],
    other_states: List[Dict[str, Any]],
    other_actions: List[int],
    episode: int,
    turn: int,
    action_logger: Optional[logging.Logger],
) -> Tuple[Dict[str, Any], int, bool, bool]:
    """
    Execute a single player's turn.
    
    Args:
        env: Game environment
        player: Current player
        other_player: Opponent player
        actions: Total number of actions
        steps: Total training steps
        validating: Whether in validation mode
        player_states: States list for current player
        player_actions: Actions list for current player
        other_states: States list for other player
        other_actions: Actions list for other player
        episode: Current episode number
        turn: Current turn number
        action_logger: Logger for action logging
        
    Returns:
        Tuple of (observation, score, terminated, truncated)
    """
    turn_start_time = time.time()
    
    # Get initial observation
    observation = env.get_obs()
    player_states.append(observation)
    
    # Capture state features IMMEDIATELY (before any actions modify the state)
    # This avoids aliasing issues where shallow copies point to modified arrays
    state_features_before = extract_state_features(observation)
    score_before, _ = env.scoreState()
    
    # Get and execute action
    valid_actions = env.generateActionMask()
    action = player.getAction(observation, valid_actions, actions, steps, force_greedy=validating)
    player_actions.append(action)
    env.updateStack(action)
    
    # Handle counter exchange
    zones_swapped = handle_counter_exchange(
        env, player, other_player, actions, steps, validating,
        player_states, player_actions, other_states, other_actions
    )
    
    # Resolve stack
    env.resolveStack()
    
    # Switch back to original player if zones are swapped
    if zones_swapped:
        env.passControl()
    
    # Execute main action if stack is not empty
    score = 0
    terminated = False
    truncated = False
    if env.stackTop() != 0:
        env.emptyStack()
        observation, score, terminated, truncated = env.step(action)
        # Capture state features AFTER action
        state_features_after = extract_state_features(observation)
        
        if action_logger:
            log_action(
                episode=episode,
                turn=turn,
                player_name=player.name,
                action_id=action,
                state_features_before=state_features_before,
                state_features_after=state_features_after,
                score_before=score_before,
                score_after=score,
                valid_actions=valid_actions,
                action_logger=action_logger,
                env=env,
                terminated=terminated,
                truncated=truncated,
            )
    
    # Handle special stack resolutions (Seven, Four)
    if not terminated and not truncated:
        obs, sc, term, trunc = handle_stack_resolution(
            env, player, other_player, actions, steps, validating,
            player_states, player_actions, other_states, other_actions,
            episode, turn, action_logger
        )
        if obs is not None:
            observation, score, terminated, truncated = obs, sc, term, trunc
    
    # Timing diagnostic for slow turns
    turn_time = time.time() - turn_start_time
    if turn_time > SLOW_TURN_THRESHOLD:
        action_obj = env.action_registry.get_action(action)
        action_type = action_obj.__class__.__name__ if action_obj else "Unknown"
        action_args = action_obj.args if action_obj else None
        print(f"[SLOW TURN] Episode {episode}, Turn {turn}: {turn_time:.3f}s | "
              f"Player: {player.name} | Action: {action_type}(id={action}, args={action_args})")
    
    return observation, score, terminated, truncated


def selfPlayTraining(
    p1: Players.Player,
    p2: Players.Player,
    episodes: int,
    validating: bool = False,
    log_actions: bool = True,
    log_metrics: bool = True,
    model_id: Optional[str] = None,
    include_highest_point_value: bool = False,
    include_highest_point_value_opponent_field: bool = False,
    initial_steps: int = 0,
    round_number: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Execute self-play training between two players.
    
    This function runs multiple episodes of gameplay between two players,
    collecting training data and logging actions/metrics as specified.
    
    Args:
        p1: First player (typically the training agent)
        p2: Second player (can be same as p1 for self-play)
        episodes: Number of episodes to train
        validating: If True, use greedy policy (no exploration)
        log_actions: If True, log all actions to file for strategy analysis
        log_metrics: If True, log training metrics (win rates, loss, etc.)
        model_id: Optional model identifier for log files (e.g., "round_3", "checkpoint_5")
        include_highest_point_value: If True, include "Highest Point Value in Hand" feature
        include_highest_point_value_opponent_field: If True, include "Highest Point Value in Opponent Field" feature
        initial_steps: Starting step count for epsilon decay (for resuming training)
        round_number: Optional round number to display in episode output
        
    Returns:
        Tuple of (p1_wins, p2_wins, final_steps) counts
    """
    env = CuttleEnvironment(
        include_highest_point_value=include_highest_point_value,
        include_highest_point_value_opponent_field=include_highest_point_value_opponent_field
    )
    actions = env.actions
    steps = initial_steps
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # Setup loggers with model identifier
    action_logger = setup_action_logger(LOG_DIRECTORY, model_id) if log_actions else None
    metrics_logger = setup_metrics_logger(LOG_DIRECTORY, model_id) if log_metrics else None
    
    for episode in range(episodes):
        episode_start_time = time.time()
        env.reset()
        p1_states: List[Dict[str, Any]] = []
        p1_actions: List[int] = []
        p2_states: List[Dict[str, Any]] = []
        p2_actions: List[int] = []
        
        turns = 0
        terminated = False
        truncated = False
        p1_score = 0
        p2_score = 0
        
        # Main game loop
        while not terminated and not truncated and turns < MAX_TURNS_PER_EPISODE:
            turns += 1
            # Only increment steps during training (not validation)
            # This ensures epsilon decay is based on actual training steps
            if not validating:
                steps += 1
            
            # Player 1's turn
            observation, p1_score, terminated, truncated = execute_player_turn(
                env, p1, p2, actions, steps, validating,
                p1_states, p1_actions, p2_states, p2_actions,
                episode, turns, action_logger
            )
            
            if terminated:
                # P1 wins
                p1_wins += 1
                if not validating:
                    update_replay_memory(p1, p1_states, p1_actions, REWARD_WIN)
                    update_replay_memory(p2, p2_states, p2_actions, REWARD_LOSS)
                break
            
            if truncated or turns >= MAX_TURNS_PER_EPISODE:
                draws += 1
                log_episode_outcome(
                    episode, turns, "draw", p1_score, p2_score,
                    "max_turns_reached" if turns >= MAX_TURNS_PER_EPISODE else "deck_exhausted",
                    action_logger
                )
                if not validating:
                    update_replay_memory(p1, p1_states, p1_actions, REWARD_DRAW)
                    update_replay_memory(p2, p2_states, p2_actions, REWARD_DRAW)
                break
            
            # End turn and prepare for P2
            env.emptyStack()
            env.passControl()
            env.end_turn()
            p2_next_state = env.get_obs()
            if not validating:
                update_replay_memory(p2, p2_states, p2_actions, REWARD_INTERMEDIATE, p2_next_state)
            p2_states = []
            p2_actions = []
            
            # Player 2's turn
            observation, p2_score, terminated, truncated = execute_player_turn(
                env, p2, p1, actions, steps, validating,
                p2_states, p2_actions, p1_states, p1_actions,
                episode, turns, action_logger
            )
            
            if terminated:
                # P2 wins
                p2_wins += 1
                if not validating:
                    update_replay_memory(p1, p1_states, p1_actions, REWARD_LOSS)
                    update_replay_memory(p2, p2_states, p2_actions, REWARD_WIN)
                break
            
            if truncated or turns >= MAX_TURNS_PER_EPISODE:
                draws += 1
                log_episode_outcome(
                    episode, turns, "draw", p1_score, p2_score,
                    "max_turns_reached" if turns >= MAX_TURNS_PER_EPISODE else "deck_exhausted",
                    action_logger
                )
                if not validating:
                    update_replay_memory(p1, p1_states, p1_actions, REWARD_DRAW)
                    update_replay_memory(p2, p2_states, p2_actions, REWARD_DRAW)
                break
            
            # End turn and prepare for next iteration
            env.emptyStack()
            env.end_turn()
            env.passControl()
            p1_next_state = env.get_obs()
            if not validating:
                update_replay_memory(p1, p1_states, p1_actions, REWARD_INTERMEDIATE, p1_next_state)
            p1_states = []
            p1_actions = []
        
        # End of episode - optimize if training
        loss = None
        if not validating:
            loss = p1.optimize()
        
        # Calculate and log metrics
        p1_win_rate = p1_wins / (episode + 1)
        p2_win_rate = p2_wins / (episode + 1)
        draw_rate = draws / (episode + 1)
        
        log_episode_metrics(
            episode, p1, p2, p1_wins, p2_wins, draws, steps, turns,
            p1_score, p2_score, validating, loss, metrics_logger
        )
        
        # Print episode summary
        episode_elapsed_time = time.time() - episode_start_time
        round_str = f"Round {round_number} " if round_number is not None else ""
        print(f"{round_str}Episode {episode}: {p1.name}: {p1_score} {p2.name}: {p2_score}")
        print(f"{p1.name}: {p1_win_rate:.3f} {p2.name}: {p2_win_rate:.3f} Draws: {draw_rate:.3f}")
        if not validating and loss is not None:
            print(f"Loss: {loss:.6f}")
        print(f"Time: {episode_elapsed_time:.2f}s")
        
        # Brief pause between episodes to reduce sustained AVX load and CPU thermal stress
        time.sleep(0.01)
    
    return p1_wins, p2_wins, steps


def validate_both_positions(
    trainee: Players.Player,
    opponent: Players.Player,
    episodes_per_position: int,
    include_highest_point_value: bool = False,
    include_highest_point_value_opponent_field: bool = False,
    model_id_prefix: Optional[str] = None,
    round_number: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Run validation with trainee in both positions (first and second player) for fair evaluation.
    
    In Cuttle, the dealer (second player) gets 6 cards vs 5 for the first player, which can create
    a positional advantage. Testing both positions gives a more accurate assessment.
    
    Args:
        trainee: The agent being evaluated
        opponent: The opponent to play against
        episodes_per_position: Number of episodes to run for each position (total = 2 * episodes_per_position)
        include_highest_point_value: If True, include "Highest Point Value in Hand" feature
        include_highest_point_value_opponent_field: If True, include "Highest Point Value in Opponent Field" feature
        model_id_prefix: Prefix for log file names (e.g., "no_features_round_0_vs_randomized")
        round_number: Optional round number to display in episode output
        
    Returns:
        Tuple of (trainee_wins, opponent_wins) across both positions
    """
    # Run with trainee as P1 (first player, 5 cards)
    p1w_as_p1, p2w_as_p1, _ = selfPlayTraining(
        trainee, opponent, episodes_per_position,
        validating=True,
        model_id=f"{model_id_prefix}_trainee_first" if model_id_prefix else None,
        include_highest_point_value=include_highest_point_value,
        include_highest_point_value_opponent_field=include_highest_point_value_opponent_field,
        initial_steps=0,
        round_number=round_number
    )
    
    # Run with trainee as P2 (second player/dealer, 6 cards)
    p2w_as_p2, p1w_as_p2, _ = selfPlayTraining(
        opponent, trainee, episodes_per_position,
        validating=True,
        model_id=f"{model_id_prefix}_trainee_second" if model_id_prefix else None,
        include_highest_point_value=include_highest_point_value,
        include_highest_point_value_opponent_field=include_highest_point_value_opponent_field,
        initial_steps=0,
        round_number=round_number
    )
    
    # Combine results: trainee wins = wins when trainee was P1 + wins when trainee was P2
    trainee_wins = p1w_as_p1 + p1w_as_p2
    opponent_wins = p2w_as_p1 + p2w_as_p2
    
    return trainee_wins, opponent_wins
