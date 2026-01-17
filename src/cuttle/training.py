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

import numpy as np
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
REWARD_INTERMEDIATE = 0.0  # Base reward for intermediate steps (non-terminal states)
SCORE_REWARD_SCALE = 0.01  # Scale factor for score-based rewards (reduced from 0.1 to prevent Q-value explosion)
GAP_REWARD_SCALE = 0.005  # Scale factor for score gap rewards (half of score reward scale to prioritize scoring over gap)


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


def extract_training_type(model_id: Optional[str]) -> Optional[str]:
    """
    Extract training type from model_id to organize logs into subdirectories.
    
    Args:
        model_id: Model identifier (e.g., "hand_only_round_0_selfplay", "opponent_field_only_round_1_vs_randomized")
        
    Returns:
        Training type string (e.g., "hand_only", "opponent_field_only") or None if not recognized
    """
    if not model_id:
        return None
    
    # Known training types
    training_types = [
        "hand_only",
        "opponent_field_only",
        "no_features",
        "both_features",  # Legacy name, kept for backward compatibility
        "all_features",
        "scores"
    ]
    
    # Check if model_id starts with any known training type
    for training_type in training_types:
        if model_id.startswith(training_type):
            return training_type
    
    return None


def setup_action_logger(log_dir: Path, model_id: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Set up action logging for strategy analysis.
    
    Logs are organized into subdirectories by training type for easy access:
    - action_logs/hand_only/
    - action_logs/opponent_field_only/
    - action_logs/no_features/
    - action_logs/both_features/
    
    Args:
        log_dir: Base directory for log files
        model_id: Optional model identifier (e.g., "hand_only_round_0_selfplay")
        
    Returns:
        Configured logger or None if logging disabled
    """
    # Extract training type and create subdirectory
    training_type = extract_training_type(model_id)
    if training_type:
        subdir = log_dir / training_type
    else:
        subdir = log_dir
    
    subdir.mkdir(parents=True, exist_ok=True)
    
    if model_id:
        log_file = subdir / f"actions_{model_id}.jsonl"
    else:
        log_file = subdir / f"actions.jsonl"
    logger = setup_logger("action_logger", log_file)
    print(f"Action logging enabled: {log_file}")
    return logger


def setup_metrics_logger(log_dir: Path, model_id: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Set up metrics logging for training statistics.
    
    Logs are organized into subdirectories by training type for easy access:
    - action_logs/hand_only/
    - action_logs/opponent_field_only/
    - action_logs/no_features/
    - action_logs/both_features/
    
    Args:
        log_dir: Base directory for log files
        model_id: Optional model identifier (e.g., "hand_only_round_0_selfplay")
        
    Returns:
        Configured logger or None if logging disabled
    """
    # Extract training type and create subdirectory
    training_type = extract_training_type(model_id)
    if training_type:
        subdir = log_dir / training_type
    else:
        subdir = log_dir
    
    subdir.mkdir(parents=True, exist_ok=True)
    
    if model_id:
        metrics_file = subdir / f"metrics_{model_id}.jsonl"
    else:
        metrics_file = subdir / f"metrics.jsonl"
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
    # Stack is now a boolean array - count cards in stack
    stack_size = int(state["Stack"].sum()) if isinstance(state["Stack"], np.ndarray) else 0
    # Note: stack_top action type is not available from observation, use 0 as placeholder
    # (actual stack top action type is tracked internally by environment)
    return {
        "hand_size": int(state["Current Zones"]["Hand"].sum()),
        "field_size": int(state["Current Zones"]["Field"].sum()),
        "opponent_field_size": int(state["Off-Player Field"].sum()),
        "deck_size": int(state["Deck"].sum()),
        "scrap_size": int(state["Scrap"].sum()),
        "stack_size": stack_size,  # Number of cards in stack (boolean array)
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
    score_change: Optional[int] = None,
    gap_change: Optional[int] = None,
) -> None:
    """
    Update player's replay memory with episode transitions.
    
    For terminal states (reward is WIN/LOSS/DRAW), all actions in the final turn
    get the terminal reward. This is appropriate because in Cuttle, a turn can include
    multiple actions (main action, counters, stack resolutions) that all contribute
    to the outcome. The reward propagates backwards through the turn via the
    Bellman equation in Q-learning.
    
    For intermediate states, rewards are based on score changes and score gap changes
    to provide more frequent learning signals. Score rewards are scaled to not overwhelm
    terminal rewards. Gap rewards encourage maintaining or improving relative position.
    
    Args:
        player: Player whose memory to update
        states: List of states from the turn (all from the same turn when terminal)
        actions: List of actions taken
        reward: Reward for the episode (terminal reward) or base intermediate reward
        next_state: Next state (if available). None indicates terminal state.
        score_change: Change in player's score during this turn (for intermediate rewards)
        gap_change: Change in score gap (player_score - opponent_score) during this turn
    """
    if len(states) == 0:
        return
    
    # If this is a terminal reward (WIN/LOSS/DRAW), all actions in the final turn
    # should get the terminal reward. This makes sense because:
    # 1. All actions in the final turn are part of the winning/losing sequence
    # 2. The reward will propagate backwards through the turn via Q-learning
    # 3. This provides stronger learning signal for the entire winning sequence
    is_terminal = (next_state is None) and (reward in [REWARD_WIN, REWARD_LOSS, REWARD_DRAW])
    
    # Calculate score-based reward for intermediate states
    score_reward = 0.0
    if score_change is not None and not is_terminal:
        # Add score-based reward: positive for scoring points, scaled to not overwhelm terminal rewards
        score_reward = score_change * SCORE_REWARD_SCALE
    
    # Calculate gap-based reward for intermediate states
    gap_reward = 0.0
    if gap_change is not None and not is_terminal:
        # Add gap-based reward: positive for improving relative position (closing gap when behind,
        # or increasing gap when ahead), scaled smaller than score rewards to prioritize scoring
        gap_reward = gap_change * GAP_REWARD_SCALE
    
    if is_terminal:
        # All states in the final turn get the terminal reward
        # Maintain proper state chain: each state points to the next, final state has next_state=None
        for i in range(len(states)):
            action_tensor = torch.tensor([actions[i]])
            reward_tensor = torch.tensor([reward])
            # Final state has next_state = None (terminal)
            # All other states point to the next state in the sequence
            if i == len(states) - 1:
                next_state_tensor = None
            else:
                next_state_tensor = states[i + 1]
            player.memory.push(states[i], action_tensor, next_state_tensor, reward_tensor)
    else:
        # Non-terminal: store all states with base reward + score-based reward + gap-based reward
        total_reward = reward + score_reward + gap_reward
        for i in range(len(states)):
            action_tensor = torch.tensor([actions[i]])
            reward_tensor = torch.tensor([total_reward])
            # For the last state, use the provided next_state (if any)
            if i == len(states) - 1:
                next_state_tensor = next_state if next_state is not None else None
            else:
                # For intermediate states, next_state is the following state
                next_state_tensor = states[i + 1]
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


def check_loss_divergence(
    loss_history: List[float],
    window_size: int = 100,
    divergence_threshold: float = 0.5,
    min_episodes: int = 200
) -> bool:
    """
    Check if loss is diverging (consistently rising) to enable early stopping.
    
    Args:
        loss_history: List of recent loss values
        window_size: Number of recent episodes to consider
        divergence_threshold: Minimum average slope to consider as divergence
        min_episodes: Minimum episodes before early stopping can trigger
        
    Returns:
        True if loss is diverging and early stopping should trigger
    """
    if len(loss_history) < min_episodes:
        return False
    
    # Get recent window of losses
    recent_losses = loss_history[-window_size:]
    
    # Check for NaN or infinite values (definite divergence)
    if any(not math.isfinite(l) for l in recent_losses):
        return True
    
    # Calculate linear trend (slope)
    x = list(range(len(recent_losses)))
    if len(x) < 2:
        return False
    
    # Simple linear regression to get slope
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(recent_losses)
    sum_xy = sum(x[i] * recent_losses[i] for i in range(n))
    sum_x2 = sum(xi * xi for xi in x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return False
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    
    # Check if slope exceeds threshold (positive = rising loss)
    return slope > divergence_threshold


def selfPlayTraining(
    p1: Players.Player,
    p2: Players.Player,
    episodes: int,
    validating: bool = False,
    log_actions: bool = True,
    log_metrics: bool = True,
    model_id: Optional[str] = None,
    initial_steps: int = 0,
    round_number: Optional[int] = None,
    initial_total_time: float = 0.0,
    early_stopping_config: Optional[Dict[str, Any]] = None,
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
        initial_steps: Starting step count for epsilon decay (for resuming training)
        round_number: Optional round number to display in episode output
        initial_total_time: Accumulated time from previous training sessions (for checkpoint resumption)
        early_stopping_config: Configuration dict for early stopping
        
    Returns:
        Tuple of (p1_wins, p2_wins, final_steps) counts
    """
    env = CuttleEnvironment()
    actions = env.actions
    steps = initial_steps
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # Setup loggers with model identifier
    action_logger = setup_action_logger(LOG_DIRECTORY, model_id) if log_actions else None
    metrics_logger = setup_metrics_logger(LOG_DIRECTORY, model_id) if log_metrics else None
    
    # Early stopping setup
    loss_history = []
    early_stopping_enabled = False
    check_interval = 50
    window_size = 100
    divergence_threshold = 0.5
    min_episodes = 200
    max_loss = 50.0
    
    if early_stopping_config is not None and early_stopping_config.get("enabled", False):
        early_stopping_enabled = True
        check_interval = early_stopping_config.get("check_interval", 50)
        window_size = early_stopping_config.get("window_size", 100)
        divergence_threshold = early_stopping_config.get("divergence_threshold", 0.5)
        min_episodes = early_stopping_config.get("min_episodes", 200)
        max_loss = early_stopping_config.get("max_loss", 50.0)
    
    # Track total time from start of training (including previous sessions if resuming)
    session_start_time = time.time()
    
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
            
            # Track scores before P1's turn for score-based rewards and gap calculation
            # We're in P1's perspective at the start of the loop
            p1_score_before, _ = env.scoreState()
            # Switch to P2's perspective to get P2's score before P1's turn
            env.passControl()
            p2_score_before_p1_turn, _ = env.scoreState()
            env.passControl()  # Switch back to P1's perspective
            
            # Calculate initial gap (P1's perspective: positive = ahead, negative = behind)
            gap_before_p1_turn = p1_score_before - p2_score_before_p1_turn
            
            # Player 1's turn
            observation, p1_score, terminated, truncated = execute_player_turn(
                env, p1, p2, actions, steps, validating,
                p1_states, p1_actions, p2_states, p2_actions,
                episode, turns, action_logger
            )
            
            # Calculate score change for P1 (p1_score is from P1's perspective after their turn)
            p1_score_change = p1_score - p1_score_before
            
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
            # Track score after P1's turn (we're now in P2's perspective after passControl)
            p2_score_after_p1_turn, _ = env.scoreState()
            p2_score_before, _ = env.scoreState()  # Same as above, for P2's upcoming turn
            if not validating:
                # Calculate P2's score change during P1's turn (e.g., if P1 scuttled P2's cards)
                p2_score_change_during_p1_turn = p2_score_after_p1_turn - p2_score_before_p1_turn
                # Calculate gap change for P2 during P1's turn
                # From P2's perspective: gap = p2_score - p1_score
                # Gap before: p2_score_before_p1_turn - p1_score_before
                # Gap after: p2_score_after_p1_turn - p1_score (p1_score is from P1's perspective after their turn)
                env.passControl()  # Switch to P1's perspective to get P1's score after their turn
                p1_score_after_turn, _ = env.scoreState()
                env.passControl()  # Switch back to P2's perspective
                gap_before_p1_turn_p2_perspective = p2_score_before_p1_turn - p1_score_before
                gap_after_p1_turn_p2_perspective = p2_score_after_p1_turn - p1_score_after_turn
                gap_change_for_p2_during_p1_turn = gap_after_p1_turn_p2_perspective - gap_before_p1_turn_p2_perspective
                update_replay_memory(p2, p2_states, p2_actions, REWARD_INTERMEDIATE, p2_next_state, p2_score_change_during_p1_turn, gap_change_for_p2_during_p1_turn)
            p2_states = []
            p2_actions = []
            
            # Player 2's turn
            observation, p2_score, terminated, truncated = execute_player_turn(
                env, p2, p1, actions, steps, validating,
                p2_states, p2_actions, p1_states, p1_actions,
                episode, turns, action_logger
            )
            
            # Calculate score change for P2 (p2_score is from P2's perspective after their turn)
            p2_score_change = p2_score - p2_score_before
            
            # Calculate gap change for P2 after their turn
            # From P2's perspective: gap = p2_score - p1_score
            # Gap before P2's turn: gap_before_p2_turn = p2_score_before - p1_score_before_p2_turn
            # Gap after P2's turn: p2_score - p1_score_after_p2_turn
            env.passControl()  # Switch to P1's perspective to get P1's score
            p1_score_before_p2_turn, _ = env.scoreState()
            env.passControl()  # Switch back to P2's perspective
            gap_before_p2_turn = p2_score_before - p1_score_before_p2_turn
            gap_after_p2_turn = p2_score - p1_score_before_p2_turn  # P1's score hasn't changed during P2's turn
            gap_change_for_p2_after_turn = gap_after_p2_turn - gap_before_p2_turn
            
            # Update P2's replay memory with their turn's results (before checking termination)
            # We need to prepare the next state for P2, which will be after the turn ends
            if not validating:
                # We'll update P2's memory after we prepare the next state
                # For now, store the gap change to use later
                pass
            
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
                # Update P2's replay memory with their turn's results
                # We calculated gap_change_for_p2_after_turn earlier, now we can use it
                # The next state for P2 is actually the state after the turn ends (which is P1's perspective)
                # But we need to get it from P2's perspective
                env.passControl()  # Switch to P2's perspective
                p2_next_state = env.get_obs()
                env.passControl()  # Switch back to P1's perspective
                update_replay_memory(p2, p2_states, p2_actions, REWARD_INTERMEDIATE, p2_next_state, p2_score_change, gap_change_for_p2_after_turn)
                
                # Calculate score change for P1 (we're back in P1's perspective)
                # p1_score_change was already calculated above, use it
                # Calculate gap change for P1 after P2's turn
                # We're in P1's perspective, so gap = p1_score - p2_score
                # Gap before P1's turn: gap_before_p1_turn (already calculated)
                # Gap after P2's turn: p1_score_after_p2_turn - p2_score_after_turn
                p1_score_after_p2_turn, _ = env.scoreState()
                env.passControl()  # Switch to P2's perspective to get P2's score after their turn
                p2_score_after_turn, _ = env.scoreState()
                env.passControl()  # Switch back to P1's perspective
                gap_after_p2_turn_p1_perspective = p1_score_after_p2_turn - p2_score_after_turn
                # Gap change for P1: how much the gap changed from before P1's turn to after P2's turn
                # This captures the net effect of both players' turns
                gap_change_for_p1_after_p2_turn = gap_after_p2_turn_p1_perspective - gap_before_p1_turn
                update_replay_memory(p1, p1_states, p1_actions, REWARD_INTERMEDIATE, p1_next_state, p1_score_change, gap_change_for_p1_after_p2_turn)
            p1_states = []
            p1_actions = []
        
        # End of episode - optimize if training
        loss = None
        if not validating:
            loss = p1.optimize()
        
        # Track loss for early stopping
        if loss is not None and early_stopping_enabled:
            loss_history.append(loss)
            
            if early_stopping_enabled:
                # Check for maximum loss threshold
                if loss > max_loss:
                    print(f"\n{'!'*60}")
                    print(f"EARLY STOPPING: Loss exceeded maximum threshold ({loss:.2f} > {max_loss})")
                    print(f"Stopping training at episode {episode + 1}/{episodes}")
                    print(f"This indicates hyperparameters may be causing divergence.")
                    print(f"{'!'*60}\n")
                    break
                
                # Check for divergence periodically
                if (episode + 1) % check_interval == 0 and len(loss_history) >= window_size:
                    if check_loss_divergence(loss_history, window_size, divergence_threshold, min_episodes):
                        recent_avg = sum(loss_history[-window_size:]) / window_size
                        print(f"\n{'!'*60}")
                        print(f"EARLY STOPPING: Loss divergence detected")
                        print(f"Recent average loss: {recent_avg:.4f}")
                        print(f"Stopping training at episode {episode + 1}/{episodes}")
                        print(f"Consider adjusting: learning_rate, target_update_frequency, or gradient_clip_norm")
                        print(f"{'!'*60}\n")
                        break
        
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
        session_elapsed_time = time.time() - session_start_time
        total_elapsed_time = initial_total_time + session_elapsed_time
        round_str = f"Round {round_number} " if round_number is not None else ""
        print(f"{round_str}Episode {episode}: {p1.name}: {p1_score} {p2.name}: {p2_score}")
        print(f"{p1.name}: {p1_win_rate:.3f} {p2.name}: {p2_win_rate:.3f} Draws: {draw_rate:.3f}")
        if not validating and loss is not None:
            print(f"Loss: {loss:.6f}")
        print(f"Time: {episode_elapsed_time:.2f}s | Total: {total_elapsed_time:.2f}s")
        
        # Brief pause between episodes to reduce sustained AVX load and CPU thermal stress
        time.sleep(0.01)
    
    return p1_wins, p2_wins, steps


def validate_both_positions(
    trainee: Players.Player,
    opponent: Players.Player,
    episodes_per_position: int,
    model_id_prefix: Optional[str] = None,
    round_number: Optional[int] = None,
    initial_total_time: float = 0.0,
) -> Tuple[int, int]:
    """
    Run validation with trainee in both positions (first and second player) for fair evaluation.
    
    In Cuttle, the dealer (second player) gets 6 cards vs 5 for the first player, which can create
    a positional advantage. Testing both positions gives a more accurate assessment.
    
    Args:
        trainee: The agent being evaluated
        opponent: The opponent to play against
        episodes_per_position: Number of episodes to run for each position (total = 2 * episodes_per_position)
        model_id_prefix: Prefix for log file names (e.g., "round_0_vs_randomized")
        round_number: Optional round number to display in episode output
        initial_total_time: Accumulated time from previous training sessions (for checkpoint resumption)
        
    Returns:
        Tuple of (trainee_wins, opponent_wins) across both positions
    """
    # Track time for first validation run
    validation_start_time = time.time()
    
    # Run with trainee as P1 (first player, 5 cards)
    p1w_as_p1, p2w_as_p1, _ = selfPlayTraining(
        trainee, opponent, episodes_per_position,
        validating=True,
        model_id=f"{model_id_prefix}_trainee_first" if model_id_prefix else None,
        initial_steps=0,
        round_number=round_number,
        initial_total_time=initial_total_time
    )
    
    # Update total time after first validation
    first_validation_time = time.time() - validation_start_time
    updated_total_time = initial_total_time + first_validation_time
    
    # Run with trainee as P2 (second player/dealer, 6 cards)
    p2w_as_p2, p1w_as_p2, _ = selfPlayTraining(
        opponent, trainee, episodes_per_position,
        validating=True,
        model_id=f"{model_id_prefix}_trainee_second" if model_id_prefix else None,
        initial_steps=0,
        round_number=round_number,
        initial_total_time=updated_total_time
    )
    
    # Combine results: trainee wins = wins when trainee was P1 + wins when trainee was P2
    trainee_wins = p1w_as_p1 + p1w_as_p2
    opponent_wins = p2w_as_p1 + p2w_as_p2
    
    return trainee_wins, opponent_wins
