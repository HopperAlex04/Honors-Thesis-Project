"""
Player classes for Cuttle game environment.
Provides base Player interface and concrete implementations for different strategies.
"""

import math
import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical


class Player(ABC):
    """Base class for all players in the Cuttle game."""
    
    def __init__(self, name: str):
        """
        Initialize a player.
        
        Args:
            name: Player's name/identifier
        """
        self.name = name

    @abstractmethod
    def getAction(
        self, 
        observation: dict, 
        valid_actions: List[int], 
        total_actions: int, 
        steps_done: int, 
        force_greedy: bool = False
    ) -> int:
        """
        Select an action given the current game state.
        
        Args:
            observation: Current game observation (dict from env.get_obs())
            valid_actions: List of valid action indices for current state
            total_actions: Total number of possible actions in the environment
            steps_done: Number of steps completed in training (for epsilon decay)
            force_greedy: If True, always use greedy policy (no exploration)
        
        Returns:
            Selected action index (integer)
        """
        pass


class Randomized(Player):
    """Player that selects actions randomly from valid actions."""
    
    def __init__(self, name: str, seed: Optional[int] = None):
        """
        Initialize a randomized player.
        
        Args:
            name: Player's name/identifier
            seed: Optional random seed for reproducibility
        """
        super().__init__(name)
        if seed is not None:
            random.seed(seed)

    def getAction(
        self, 
        observation: dict, 
        valid_actions: List[int], 
        total_actions: int, 
        steps_done: int, 
        force_greedy: bool = False
    ) -> int:
        """
        Select a random action from valid actions.
        
        Args:
            observation: Not used by this player
            valid_actions: List of valid action indices
            total_actions: Not used by this player
            steps_done: Not used by this player
            force_greedy: Not used by this player
        
        Returns:
            Randomly selected action index from valid_actions
        """
        if not valid_actions:
            return 0
        return random.choice(valid_actions)


class HeuristicHighCard(Player):
    """Heuristic player that prefers scoring high-value cards."""
    
    def getAction(
        self, 
        observation: dict, 
        valid_actions: List[int], 
        total_actions: int, 
        steps_done: int, 
        force_greedy: bool = False
    ) -> int:
        """
        Select action by preferring to score high-value cards.
        
        Args:
            observation: Not used by this player
            valid_actions: List of valid action indices
            total_actions: Not used by this player
            steps_done: Not used by this player
            force_greedy: Not used by this player
        
        Returns:
            Action index for scoring highest valid card, or 0 if none found
        """
        # Prefer actions that score cards (actions 1-48 are score actions for cards 0-51, excluding Jacks)
        # This is a simple heuristic: prefer scoring actions over other actions
        act_out = 0
        for action_id in valid_actions:
            # Score actions are typically in range 1-48 (after draw action at 0)
            # This is a simplified heuristic - in practice, you'd want to check action types
            if 1 <= action_id <= 48:
                act_out = action_id
        
        # If no score action found, return first valid action or 0
        return act_out if act_out > 0 else (valid_actions[0] if valid_actions else 0)


class ScoreGapMaximizer(Player):
    """
    Player that selects actions to maximize the score gap (player_score - opponent_score).
    
    This player evaluates each valid action by estimating the change in score gap
    that would result from taking that action, then selects the action that
    maximizes this gap.
    """
    
    def __init__(self, name: str):
        """
        Initialize a score gap maximizer player.
        
        Args:
            name: Player's name/identifier
        """
        super().__init__(name)
        # Create action registry once (it's static and doesn't depend on game state)
        from cuttle.environment import CuttleEnvironment
        temp_env = CuttleEnvironment()
        self.action_registry = temp_env.action_registry
    
    def _calculate_card_value(self, card_index: int, card_dict: dict) -> int:
        """
        Calculate the point value of a card when scored.
        
        Args:
            card_index: Index of the card (0-51)
            card_dict: Dictionary mapping card indices to rank/suit info
            
        Returns:
            Point value of the card (0 for Jacks, rank+1 for others, special for Kings)
        """
        card_info = card_dict.get(card_index, {})
        rank = card_info.get("rank", 0)
        
        # Jacks (rank 10) cannot be scored, return 0
        if rank == 10:
            return 0
        
        # Kings (rank 12) don't add points but reduce threshold
        # We'll give them a high value since they're very valuable
        if rank == 12:
            return 20  # High value for threshold reduction
        
        # Queens (rank 11) don't add points directly
        if rank == 11:
            return 5  # Moderate value for queen effects
        
        # All other cards: rank + 1 points
        return rank + 1
    
    def _estimate_score_gap_change(
        self, 
        action_obj, 
        observation: dict,
        action_registry
    ) -> float:
        """
        Estimate the change in score gap from taking an action.
        
        Args:
            action_obj: Action object to evaluate
            observation: Current game observation
            action_registry: Action registry for card information
            
        Returns:
            Estimated change in score gap (positive = good for us)
        """
        from cuttle.actions import ScoreAction, ScuttleAction, AceAction
        
        card_dict = action_registry.card_dict
        current_field = observation["Current Zones"]["Field"]
        opponent_field = observation["Off-Player Field"]
        
        gap_change = 0.0
        
        if isinstance(action_obj, ScoreAction):
            # Scoring a card: adds points to our score
            card = action_obj.card
            card_value = self._calculate_card_value(card, card_dict)
            gap_change = card_value
        
        elif isinstance(action_obj, ScuttleAction):
            # Scuttling: removes opponent's card (good) but uses our card (bad)
            our_card = action_obj.card
            opponent_card = action_obj.target
            
            # Value we lose from using our card
            our_card_value = self._calculate_card_value(our_card, card_dict)
            
            # Value opponent loses from losing their card
            opponent_card_value = self._calculate_card_value(opponent_card, card_dict)
            
            # Net change: opponent loses more than we lose = positive gap change
            gap_change = opponent_card_value - our_card_value
        
        elif isinstance(action_obj, AceAction):
            # Ace wipes all point cards from both fields
            # Calculate total points on both fields that would be destroyed
            our_points_lost = 0
            opponent_points_lost = 0
            
            point_indicies = action_registry.point_indicies
            for rank_list in point_indicies:
                for card_idx in rank_list:
                    card_value = self._calculate_card_value(card_idx, card_dict)
                    if current_field[card_idx]:
                        our_points_lost += card_value
                    if opponent_field[card_idx]:
                        opponent_points_lost += card_value
            
            # Net change: if opponent loses more, it's good for us
            gap_change = opponent_points_lost - our_points_lost
        
        else:
            # Other actions (Draw, Two, Three, Five, Six, etc.) - use heuristics
            from cuttle.actions import (
                DrawAction, TwoAction, ThreeAction, FiveAction, 
                SixAction, SevenAction01, NineAction
            )
            
            if isinstance(action_obj, DrawAction):
                # Draw: potentially good (more options), but no immediate score change
                gap_change = 0.1  # Small positive value for drawing
            
            elif isinstance(action_obj, TwoAction):
                # Two: Scraps opponent's royal card - good if opponent has valuable royals
                target = action_obj.target
                target_value = self._calculate_card_value(target, card_dict)
                # If opponent has this royal on field, removing it is valuable
                if opponent_field[target]:
                    gap_change = target_value * 0.5  # Moderate value
                else:
                    gap_change = 0.0  # No immediate benefit
            
            elif isinstance(action_obj, ThreeAction):
                # Three: Grabs card from scrap - could be valuable
                gap_change = 2.0  # Moderate positive value
            
            elif isinstance(action_obj, FiveAction):
                # Five: Draws 2 cards - good for options
                gap_change = 0.2  # Small positive value
            
            elif isinstance(action_obj, SixAction):
                # Six: Wipes all royal cards - calculate value
                our_royals_lost = 0
                opponent_royals_lost = 0
                
                royal_indicies = action_registry.royal_indicies
                for royal_list in royal_indicies:
                    for card_idx in royal_list:
                        card_value = self._calculate_card_value(card_idx, card_dict)
                        if current_field[card_idx]:
                            our_royals_lost += card_value
                        if opponent_field[card_idx]:
                            opponent_royals_lost += card_value
                
                gap_change = opponent_royals_lost - our_royals_lost
            
            elif isinstance(action_obj, (SevenAction01, NineAction)):
                # Seven and Nine have complex effects - moderate value
                gap_change = 1.0
            
            else:
                # Unknown action type - neutral
                gap_change = 0.0
        
        return gap_change
    
    def getAction(
        self,
        observation: dict,
        valid_actions: List[int],
        total_actions: int,
        steps_done: int,
        force_greedy: bool = False
    ) -> int:
        """
        Select action that maximizes score gap.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid action indices
            total_actions: Total number of possible actions
            steps_done: Not used by this player
            force_greedy: Not used by this player
            
        Returns:
            Action index that maximizes expected score gap
        """
        if not valid_actions:
            return 0
        
        # Evaluate each valid action
        best_action = valid_actions[0]
        best_gap_change = float("-inf")
        
        for action_id in valid_actions:
            action_obj = self.action_registry.get_action(action_id)
            if action_obj:
                gap_change = self._estimate_score_gap_change(
                    action_obj, observation, self.action_registry
                )
                
                if gap_change > best_gap_change:
                    best_gap_change = gap_change
                    best_action = action_id
        
        return best_action


class TempoPlayer(Player):
    """
    Less aggressive opponent: maximizes score gap but values tempo and avoids
    over-committing. Prefers scoring and drawing over risky Scuttle/Ace when
    the gap gain is similar; penalizes using high-value cards for scuttle and
    Acing when we have a lot on board.
    """

    def __init__(
        self,
        name: str,
        scuttle_penalty: float = 0.5,
        ace_penalty: float = 0.5,
        draw_bonus: float = 0.4,
        prefer_draw_when_close: bool = True,
        draw_tolerance: float = 0.5,
    ):
        super().__init__(name)
        from cuttle.environment import CuttleEnvironment
        temp_env = CuttleEnvironment()
        self.action_registry = temp_env.action_registry
        self.scuttle_penalty = scuttle_penalty  # subtract this * our_card_value for Scuttle
        self.ace_penalty = ace_penalty          # subtract this * our_points_lost for Ace
        self.draw_bonus = draw_bonus            # add to Draw's score so Draw is more attractive
        self.prefer_draw_when_close = prefer_draw_when_close
        self.draw_tolerance = draw_tolerance

    def _calculate_card_value(self, card_index: int, card_dict: dict) -> int:
        """Same as ScoreGapMaximizer: point value of a card when scored."""
        card_info = card_dict.get(card_index, {})
        rank = card_info.get("rank", 0)
        if rank == 10:
            return 0
        if rank == 12:
            return 20
        if rank == 11:
            return 5
        return rank + 1

    def _adjusted_gap_change(
        self,
        action_obj,
        observation: dict,
        gap_change: float,
    ) -> float:
        """Apply tempo/risk adjustments to raw gap change."""
        from cuttle.actions import ScoreAction, ScuttleAction, AceAction, DrawAction

        card_dict = self.action_registry.card_dict
        current_field = observation["Current Zones"]["Field"]
        opponent_field = observation["Off-Player Field"]

        if isinstance(action_obj, DrawAction):
            return gap_change + self.draw_bonus

        if isinstance(action_obj, ScuttleAction):
            our_val = self._calculate_card_value(action_obj.card, card_dict)
            return gap_change - self.scuttle_penalty * our_val

        if isinstance(action_obj, AceAction):
            our_points_lost = 0
            point_indicies = self.action_registry.point_indicies
            for rank_list in point_indicies:
                for card_idx in rank_list:
                    if current_field[card_idx]:
                        our_points_lost += self._calculate_card_value(card_idx, card_dict)
            return gap_change - self.ace_penalty * our_points_lost

        return gap_change

    def _estimate_score_gap_change(self, action_obj, observation: dict, action_registry) -> float:
        """Reuse ScoreGapMaximizer logic for raw gap estimate."""
        from cuttle.actions import ScoreAction, ScuttleAction, AceAction

        card_dict = action_registry.card_dict
        current_field = observation["Current Zones"]["Field"]
        opponent_field = observation["Off-Player Field"]
        gap_change = 0.0

        if isinstance(action_obj, ScoreAction):
            card_value = self._calculate_card_value(action_obj.card, card_dict)
            gap_change = card_value
        elif isinstance(action_obj, ScuttleAction):
            our_val = self._calculate_card_value(action_obj.card, card_dict)
            opp_val = self._calculate_card_value(action_obj.target, card_dict)
            gap_change = opp_val - our_val
        elif isinstance(action_obj, AceAction):
            our_pts, opp_pts = 0, 0
            for rank_list in action_registry.point_indicies:
                for card_idx in rank_list:
                    v = self._calculate_card_value(card_idx, card_dict)
                    if current_field[card_idx]:
                        our_pts += v
                    if opponent_field[card_idx]:
                        opp_pts += v
            gap_change = opp_pts - our_pts
        else:
            from cuttle.actions import (
                DrawAction, TwoAction, ThreeAction, FiveAction,
                SixAction, SevenAction01, NineAction,
            )
            if isinstance(action_obj, DrawAction):
                gap_change = 0.1
            elif isinstance(action_obj, TwoAction):
                target = action_obj.target
                v = self._calculate_card_value(target, card_dict)
                gap_change = v * 0.5 if opponent_field[target] else 0.0
            elif isinstance(action_obj, ThreeAction):
                gap_change = 2.0
            elif isinstance(action_obj, FiveAction):
                gap_change = 0.2
            elif isinstance(action_obj, SixAction):
                our_royals, opp_royals = 0, 0
                for royal_list in action_registry.royal_indicies:
                    for card_idx in royal_list:
                        v = self._calculate_card_value(card_idx, card_dict)
                        if current_field[card_idx]:
                            our_royals += v
                        if opponent_field[card_idx]:
                            opp_royals += v
                gap_change = opp_royals - our_royals
            elif isinstance(action_obj, (SevenAction01, NineAction)):
                gap_change = 1.0
            else:
                gap_change = 0.0
        return gap_change

    def getAction(
        self,
        observation: dict,
        valid_actions: List[int],
        total_actions: int,
        steps_done: int,
        force_greedy: bool = False,
    ) -> int:
        if not valid_actions:
            return 0

        draw_action_id = 0  # Draw is typically action 0
        best_action = valid_actions[0]
        best_adjusted = float("-inf")
        draw_adjusted = None

        for action_id in valid_actions:
            action_obj = self.action_registry.get_action(action_id)
            if not action_obj:
                continue
            raw = self._estimate_score_gap_change(
                action_obj, observation, self.action_registry
            )
            adjusted = self._adjusted_gap_change(action_obj, observation, raw)
            if action_id == draw_action_id:
                draw_adjusted = adjusted
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_action = action_id

        if (
            self.prefer_draw_when_close
            and draw_adjusted is not None
            and draw_action_id in valid_actions
            and best_adjusted - draw_adjusted <= self.draw_tolerance
            and best_action != draw_action_id
        ):
            return draw_action_id
        return best_action


class Agent(Player):
    """DQN-based agent that learns to play using deep reinforcement learning."""
    
    def __init__(
        self, 
        name: str, 
        model: torch.nn.Module, 
        batch_size: int,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        tau: float,
        lr: float,
        replay_buffer_size: int = 100000,
        use_prioritized_replay: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_increment: Optional[float] = None,
        per_epsilon: float = 1e-6,
        use_double_dqn: bool = False,
        q_value_clip: float = 100.0,
    ):
        """
        Initialize a DQN agent.
        
        Args:
            name: Agent's name/identifier
            model: Neural network model for Q-value estimation
            batch_size: Batch size for training
            gamma: Discount factor for future rewards
            eps_start: Initial epsilon value for epsilon-greedy exploration
            eps_end: Final epsilon value for epsilon-greedy exploration
            eps_decay: Steps over which epsilon decays from start to end
            tau: Soft update parameter (currently unused, reserved for future use)
            lr: Learning rate for optimizer
            replay_buffer_size: Size of replay memory buffer (default: 100000)
            use_prioritized_replay: If True, use Prioritized Experience Replay (PER).
            per_alpha: PER priority exponent (0 = uniform, 1 = full priority). Default 0.6.
            per_beta: PER importance-sampling exponent (start). Default 0.4.
            per_beta_end: PER importance-sampling exponent (end). Default 1.0.
            per_beta_increment: PER beta increment per sample (None = no annealing).
            per_epsilon: Small constant added to priorities to avoid zero. Default 1e-6.
            use_double_dqn: If True, use Double DQN: policy selects next action, target evaluates it.
            q_value_clip: Clip Q-values and targets to [-q_value_clip, q_value_clip] to prevent extremes. Default 100.0.
        """
        super().__init__(name)
        
        # Set up policy model
        self.model = model
        self.policy = model
        
        # Create target network for stable Q-learning (copy of policy network)
        # Import here to avoid circular dependency
        from cuttle.networks import NeuralNetwork
        import copy
        # Create a new network with same architecture
        self.target = copy.deepcopy(model)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()  # Target network is always in eval mode

        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.update_target_counter = 0  # Track steps for target network updates
        self.target_update_frequency = 0  # Hard update target network every N steps (0 = use soft updates with tau)
        self.exploration_boost = 0  # Temporary boost to exploration (subtracted from steps_done)

        # Prioritized Experience Replay (toggleable)
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(
                capacity=replay_buffer_size,
                alpha=per_alpha,
                beta=per_beta,
                beta_end=per_beta_end,
                beta_increment=per_beta_increment,
                epsilon=per_epsilon,
            )
        else:
            # Replay Memory (uniform or mix_old_new sampling)
            self.memory = ReplayMemory(replay_buffer_size)

        # Double DQN: policy selects next action, target evaluates it (reduces overestimation)
        self.use_double_dqn = use_double_dqn

        # Q-value clipping: prevents pathological values; configurable via hyperparams
        self.q_value_clip = q_value_clip

        # Using Adam optimization with weight decay to prevent catastrophic forgetting
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=1e-5  # L2 regularization to prevent large weight changes
        )

        # Using Huber Loss (Smooth L1 Loss)
        self.criterion = torch.nn.SmoothL1Loss()

    def get_optimizer_state(self) -> dict:
        """Get optimizer state dict for checkpointing."""
        return self.optimizer.state_dict()

    def set_optimizer_state(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        self.optimizer.load_state_dict(state_dict)
    
    def set_target_update_frequency(self, frequency: int) -> None:
        """
        Set target network update frequency.
        
        Args:
            frequency: Number of steps between hard updates (0 = use soft updates with tau)
        """
        self.target_update_frequency = frequency
    
    def boost_exploration(self, boost_steps: int = 0) -> None:
        """
        Temporarily boost exploration by reducing effective steps_done for epsilon calculation.
        This helps escape local minima after regression.
        
        Args:
            boost_steps: Number of steps to subtract from steps_done for epsilon calculation
                        (positive value = more exploration)
        """
        self.exploration_boost = boost_steps
    
    def reset_exploration_boost(self) -> None:
        """Reset exploration boost to normal."""
        self.exploration_boost = 0

    def getAction(
        self, 
        observation: dict, 
        valid_actions: List[int], 
        total_actions: int, 
        steps_done: int, 
        force_greedy: bool = False
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current game observation (dict from env.get_obs())
            valid_actions: List of valid action indices for current state
            total_actions: Total number of possible actions in the environment
            steps_done: Number of steps completed in training (for epsilon decay)
            force_greedy: If True, always use greedy policy (no exploration)
        
        Returns:
            Selected action index (integer)
        """
        sample = random.random()
        
        # Calculate epsilon threshold for epsilon-greedy exploration
        # As training continues, epsilon decreases, making agent more likely to exploit
        # Apply exploration boost if set (reduces effective steps_done, increasing exploration)
        effective_steps = max(0, steps_done - self.exploration_boost)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * effective_steps / self.eps_decay
        )

        # Use policy (exploit) if random sample exceeds threshold or forced greedy
        if sample > eps_threshold or force_greedy:
            with torch.no_grad():
                # Get Q-values for all actions from the policy network
                q_values = self.policy(observation)
                
                # Don't clip Q-values during action selection - let the network output natural values
                # Clipping here can cause issues with action selection and creates inconsistency
                # with training where we don't clip current Q-values
                
                # Mask invalid actions by setting their Q-values to negative infinity
                for action_id in range(total_actions):
                    if action_id not in valid_actions:
                        q_values[action_id] = float("-inf")
                
                # Return action with highest Q-value among valid actions
                return q_values.argmax().item()
        else:
            # Random exploration: select random valid action
            if not valid_actions:
                return 0
            return random.choice(valid_actions)

    def optimize(self) -> Optional[float]:
        """
        Perform one optimization step using a batch of experiences from replay memory.
        
        Uses DQN algorithm:
        1. Sample a batch of transitions from replay memory
        2. Compute current Q-values for state-action pairs
        3. Compute target Q-values using Bellman equation
        4. Compute loss and backpropagate
        5. Update network weights
        
        Returns:
            Loss value if optimization was performed, None if insufficient samples
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of transitions from replay memory
        if self.use_prioritized_replay:
            transitions, batch_indices, is_weights = self.memory.sample(self.batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)
            batch_indices = None
            is_weights = None

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Separate final and non-final states
        # Final states have next_state = None (episode ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            dtype=torch.bool
        )
        non_final_next_states = [s for s in batch.next_state if s is not None]

        # Prepare batches
        state_batch = batch.state
        action_batch = torch.cat(batch.action)  # Shape: [batch_size]
        reward_batch = torch.cat(batch.reward)  # Shape: [batch_size]

        # Compute Q(s, a) for current state-action pairs
        # unsqueeze(1) creates [batch_size, 1] shape needed for gather(1, ...)
        state_action_values = self.policy(state_batch).gather(
            1, 
            action_batch.unsqueeze(1)
        )  # Shape: [batch_size, 1]
        
        # Compute Q(s', a') for next states (for non-final states only)
        # Use target network for stability
        # CRITICAL FIX: Ensure tensor is on same device as model and has correct dtype
        device = next(self.policy.parameters()).device
        next_state_values = torch.zeros(self.batch_size, device=device, dtype=reward_batch.dtype)
        
        # Q-value clipping: configurable via q_value_clip; prevents pathological values
        q_clip = self.q_value_clip

        with torch.no_grad():
            if len(non_final_next_states) > 0:
                if self.use_double_dqn:
                    # Double DQN: policy selects best next action, target evaluates that action's Q-value
                    next_actions = self.policy(non_final_next_states).argmax(1)  # [n_non_final]
                    next_state_q_values = self.target(non_final_next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                else:
                    # Standard DQN: target network for both selection and evaluation
                    next_state_q_values = self.target(non_final_next_states).max(1).values
                next_state_q_values = torch.clamp(next_state_q_values, -q_clip, q_clip)
                next_state_values[non_final_mask] = next_state_q_values
        
        # Compute expected Q-values using Bellman equation: Q(s,a) = r + γ * max Q(s',a')
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch  # Shape: [batch_size]
        expected_state_action_values = expected_state_action_values.unsqueeze(1)  # Shape: [batch_size, 1]
        
        expected_state_action_values = torch.clamp(expected_state_action_values, -q_clip, q_clip)
        
        # CRITICAL FIX: DO NOT clip current Q-values before loss calculation
        # Clipping current Q-values creates a mismatch where the network learns values
        # that get clipped away, causing loss to continuously increase as it tries to
        # match clipped targets with values that keep exceeding the clip range.
        # Instead, let the network learn freely and only clip targets.
        # The Huber loss will naturally handle outliers without causing divergence.

        # Compute loss (element-wise for PER so we can weight and get TD errors)
        if self.use_prioritized_replay:
            device = next(self.policy.parameters()).device
            element_losses = torch.nn.functional.smooth_l1_loss(
                state_action_values, expected_state_action_values, reduction="none"
            )  # [batch_size, 1]
            weights = torch.from_numpy(is_weights).to(device=device, dtype=element_losses.dtype)
            weights = weights.unsqueeze(1)  # [batch_size, 1]
            loss = (weights * element_losses).mean()
            # TD errors for priority update (before backward)
            with torch.no_grad():
                td_errors = (
                    expected_state_action_values - state_action_values
                ).abs().squeeze(1).cpu().numpy()
        else:
            loss = self.criterion(state_action_values, expected_state_action_values)

        # Diagnostics for logging (before backward to keep graph)
        with torch.no_grad():
            q_mean = state_action_values.mean().item()
            q_std = state_action_values.std().item()
            q_min = state_action_values.min().item()
            q_max = state_action_values.max().item()
            target_mean = expected_state_action_values.mean().item()
            td_errors = expected_state_action_values - state_action_values
            td_mean = td_errors.mean().item()
            td_abs_mean = td_errors.abs().mean().item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping; returns total norm before clipping (for logging)
        grad_clip_norm = 5.0
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), grad_clip_norm
        )
        grad_clipped = total_grad_norm > grad_clip_norm

        self.optimizer.step()
        
        # Update target network
        # CRITICAL: Only increment counter when optimization actually happens
        # (we're already past the early return, so optimization ran)
        self.update_target_counter += 1
        
        if self.target_update_frequency > 0:
            # Hard update: copy policy network to target network every N steps
            if self.update_target_counter % self.target_update_frequency == 0:
                self.target.load_state_dict(self.policy.state_dict())
                # Debug: Uncomment to verify updates are happening
                # print(f"Target network updated at step {self.update_target_counter}")
        elif self.tau > 0:
            # Soft update: target = tau * policy + (1 - tau) * target (every step)
            # This keeps target network stable while gradually tracking policy updates
            for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

        # Update PER priorities after optimization (so next sample reflects new errors)
        if self.use_prioritized_replay and batch_indices is not None:
            td_errors_np = td_errors.abs().squeeze().cpu().numpy()
            self.memory.update_priorities(batch_indices, td_errors_np)

        return {
            "loss": loss.item(),
            "dqn_q_mean": q_mean,
            "dqn_q_std": q_std,
            "dqn_q_min": q_min,
            "dqn_q_max": q_max,
            "dqn_target_mean": target_mean,
            "dqn_td_mean": td_mean,
            "dqn_td_abs_mean": td_abs_mean,
            "dqn_grad_norm": total_grad_norm.item(),
            "dqn_grad_clipped": grad_clipped,
        }


class PPOAgent(Player):
    """
    Bare-minimum PPO agent: collects (s, a, log_prob, value, reward) per step,
    then runs clipped surrogate + value loss at end of episode.
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,  # ActorCritic
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        super().__init__(name)
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Per-turn lists filled by getAction; consumed by store_ppo_trajectory
        self.current_turn_log_probs: List[torch.Tensor] = []
        self.current_turn_values: List[torch.Tensor] = []
        self.current_turn_valid_actions: List[List[int]] = []  # valid action indices per step (for masking in optimize)
        # Trajectory: list of (states, actions, log_probs, values, reward, valid_actions_per_step)
        self._trajectory: List[Tuple[List[Dict], List[int], List[torch.Tensor], List[torch.Tensor], float, List[List[int]]]] = []
        self.last_log_prob: Optional[torch.Tensor] = None
        self.last_value: Optional[torch.Tensor] = None

    def _clear_turn_aux(self) -> None:
        self.current_turn_log_probs.clear()
        self.current_turn_values.clear()
        self.current_turn_valid_actions.clear()
        self.last_log_prob = None
        self.last_value = None

    def getAction(
        self,
        observation: dict,
        valid_actions: List[int],
        total_actions: int,
        steps_done: int,
        force_greedy: bool = False,
    ) -> int:
        with torch.no_grad():
            logits, value = self.model(observation)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                value = value.unsqueeze(0)
            # Mask invalid actions
            mask = torch.full((1, total_actions), float("-inf"), device=logits.device, dtype=logits.dtype)
            for a in valid_actions:
                mask[0, a] = 0
            logits = logits + mask
            dist = Categorical(logits=logits)
            action = dist.sample() if not force_greedy else logits.argmax(dim=1)
            action = action.item()
            log_prob = dist.log_prob(torch.tensor([action], device=logits.device)).squeeze(0)
            val = value.squeeze(0)
        self.last_log_prob = log_prob.detach()
        self.last_value = val.detach()
        self.current_turn_log_probs.append(log_prob.detach())
        self.current_turn_values.append(val.detach())
        self.current_turn_valid_actions.append(list(valid_actions))
        return action

    def store_ppo_trajectory(
        self,
        states: List[dict],
        actions: List[int],
        reward: float,
        log_probs: Optional[List[torch.Tensor]] = None,
        values: Optional[List[torch.Tensor]] = None,
        valid_actions_per_step: Optional[List[List[int]]] = None,
    ) -> None:
        log_probs = log_probs if log_probs is not None else self.current_turn_log_probs
        values = values if values is not None else self.current_turn_values
        valid_list = valid_actions_per_step if valid_actions_per_step is not None else self.current_turn_valid_actions
        if len(states) != len(actions) or len(states) != len(log_probs) or len(states) != len(values) or len(states) != len(valid_list):
            return
        # Store copies so clearing current_turn_* at start of next turn doesn't empty these
        self._trajectory.append((
            list(states),
            list(actions),
            list(log_probs),
            list(values),
            reward,
            [list(v) for v in valid_list],
        ))

    def optimize(self) -> Optional[float]:
        if not self._trajectory:
            return None
        # Chunk boundaries and rewards (for overwriting returns so each turn gets a single return)
        chunk_info = []  # (start_idx, length, reward) per chunk

        # Flatten trajectory. Reward: only the last step of each chunk gets that chunk's
        # reward (avoids compounding). We then overwrite returns so all steps in a chunk
        # get the same return (terminal for final turn, intermediate for other turns).
        all_states, all_actions, all_log_probs, all_values, all_rewards, all_valid_actions = [], [], [], [], [], []
        for chunk_idx, chunk in enumerate(self._trajectory):
            states, actions, log_probs, values, reward = chunk[:5]
            valid_per_step = chunk[5] if len(chunk) > 5 else [[]] * len(states)  # backward compat
            start = len(all_states)
            for i in range(len(states)):
                all_states.append(states[i])
                all_actions.append(actions[i])
                all_log_probs.append(log_probs[i])
                all_values.append(values[i])
                all_valid_actions.append(valid_per_step[i] if i < len(valid_per_step) else [])
                if i < len(states) - 1:
                    all_rewards.append(0.0)
                else:
                    all_rewards.append(reward)
            chunk_info.append((start, len(states), reward))
        self._trajectory.clear()
        self._clear_turn_aux()

        # Compute returns (Monte Carlo, no GAE)
        returns = []
        R = 0.0
        for r in reversed(all_rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns = list(reversed(returns))

        # Overwrite: all steps in each chunk get that chunk's return (dense intermediate
        # rewards + terminal on final turn). Final turn gets terminal_reward; other
        # chunks get their intermediate reward (score/gap from training loop).
        for start, length, reward in chunk_info:
            for i in range(start, min(start + length, len(returns))):
                returns[i] = reward

        device = next(self.model.parameters()).device
        returns_t = torch.tensor(returns, device=device, dtype=torch.float32)

        # Stack for batch
        old_log_probs = torch.stack([lp.to(device) for lp in all_log_probs])
        old_values = torch.stack([v.to(device) for v in all_values])
        actions_t = torch.tensor(all_actions, device=device, dtype=torch.long)

        # Advantages (simple: return - value)
        advantages_raw = returns_t - old_values.squeeze(-1)
        std = advantages_raw.std()
        if std > 1e-8 and advantages_raw.numel() > 1:
            advantages = (advantages_raw - advantages_raw.mean()) / (std + 1e-8)
        else:
            advantages = advantages_raw - advantages_raw.mean()

        total_loss_avg = 0.0
        n_batches = 0
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_ratio = None
        num_actions = self.model.num_actions
        for _ in range(self.ppo_epochs):
            logits, values = self.model(all_states)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if values.dim() == 1:
                values = values.unsqueeze(1)
            # Apply same invalid-action mask as at collection so π_new and π_old are comparable
            mask = torch.full_like(logits, float("-inf"), device=logits.device, dtype=logits.dtype)
            for i, valid in enumerate(all_valid_actions):
                if valid:
                    for a in valid:
                        mask[i, a] = 0
                else:
                    mask[i, :] = 0  # no mask if we didn't store valid (backward compat)
            logits = logits + mask
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns_t)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss_avg += loss.item()
            n_batches += 1
            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = entropy.item()
            last_ratio = ratio.detach()

        avg_loss = total_loss_avg / max(n_batches, 1)
        # Build metrics dict for logging (all scalars for JSON)
        clip_fraction = (
            (last_ratio < 1.0 - self.clip_eps).logical_or(last_ratio > 1.0 + self.clip_eps)
            .float().mean().item()
        ) if last_ratio is not None else 0.0
        # Ratio stats for debugging (ratio = π_new(a|s) / π_old(a|s); should stay near 1.0)
        ratio_mean = last_ratio.mean().item() if last_ratio is not None else 0.0
        ratio_min = last_ratio.min().item() if last_ratio is not None else 0.0
        ratio_max = last_ratio.max().item() if last_ratio is not None else 0.0
        ratio_std = last_ratio.std().item() if last_ratio is not None and last_ratio.numel() > 1 else 0.0
        return {
            "loss": avg_loss,
            "ppo_policy_loss": last_policy_loss,
            "ppo_value_loss": last_value_loss,
            "ppo_entropy": last_entropy,
            "ppo_n_steps": len(all_states),
            "ppo_n_chunks": len(chunk_info),
            "ppo_advantage_mean": advantages_raw.mean().item(),
            "ppo_advantage_std": advantages_raw.std().item(),
            "ppo_ratio_mean": ratio_mean,
            "ppo_ratio_min": ratio_min,
            "ppo_ratio_max": ratio_max,
            "ppo_ratio_std": ratio_std,
            "ppo_clip_fraction": clip_fraction,
        }

    def get_optimizer_state(self) -> dict:
        return self.optimizer.state_dict()

    def set_optimizer_state(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)


# Named tuple for storing transitions in replay memory
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """
        Save a transition to replay memory.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation (None if terminal)
            reward: Reward received
        """
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int, mix_old_new: bool = True) -> List[Transition]:
        """
        Sample a batch of transitions from replay memory.
        
        Args:
            batch_size: Number of transitions to sample
            mix_old_new: If True, sample 50% from recent experiences and 50% from older experiences
                        This helps prevent catastrophic forgetting by ensuring the agent sees both
                        old and new strategies.
            
        Returns:
            List of Transition tuples
        """
        if not mix_old_new or len(self.memory) < batch_size:
            return random.sample(self.memory, batch_size)
        
        # Mix old and new experiences to prevent forgetting
        # Recent experiences: last 20% of buffer
        # Older experiences: first 80% of buffer
        recent_start = int(len(self.memory) * 0.8)
        recent_experiences = list(self.memory)[recent_start:]
        older_experiences = list(self.memory)[:recent_start]
        
        # Sample 50% from recent, 50% from older
        recent_batch_size = batch_size // 2
        older_batch_size = batch_size - recent_batch_size
        
        sampled = []
        if len(recent_experiences) >= recent_batch_size:
            sampled.extend(random.sample(recent_experiences, recent_batch_size))
        else:
            sampled.extend(recent_experiences)
        
        if len(older_experiences) >= older_batch_size:
            sampled.extend(random.sample(older_experiences, older_batch_size))
        else:
            sampled.extend(random.sample(older_experiences, min(older_batch_size, len(older_experiences))))
        
        # If we don't have enough samples, fill with random samples from entire buffer
        if len(sampled) < batch_size:
            remaining = batch_size - len(sampled)
            sampled.extend(random.sample(self.memory, remaining))
        
        # Shuffle to mix old and new experiences
        random.shuffle(sampled)
        return sampled

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.memory)
    
    def get_state(self):
        """
        Get a copy of the current memory state for checkpointing.
        
        Returns:
            List of transitions (can be used to restore memory)
        """
        return list(self.memory)
    
    def set_state(self, memory_state):
        """
        Restore memory from a saved state.
        
        Args:
            memory_state: List of transitions from get_state()
        """
        self.memory = deque(memory_state, maxlen=self.memory.maxlen)


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) buffer.
    Samples transitions with probability proportional to priority^alpha,
    and uses importance-sampling weights (beta) in the loss to correct bias.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_increment: Optional[float] = None,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            capacity: Maximum number of transitions.
            alpha: Priority exponent (0 = uniform, 1 = full priority). Typical 0.6.
            beta: Importance-sampling exponent start. Typical 0.4.
            beta_end: Importance-sampling exponent end (annealing). Typical 1.0.
            beta_increment: Increase beta by this much per sample (None = no annealing).
            epsilon: Small constant added to priorities so none are zero.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_increment = beta_increment if beta_increment is not None else (beta_end - beta) / 1e6
        self.epsilon = epsilon
        self._memory: List[Transition] = []
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._position = 0
        self._size = 0

    def push(self, state, action, next_state, reward):
        """Store a transition with priority = max(priorities) or 1.0 so new transitions get sampled."""
        transition = Transition(state, action, next_state, reward)
        if self._size < self.capacity:
            self._memory.append(transition)
            self._size += 1
            max_prio = self._priorities[: self._size].max() if self._size > 0 else 1.0
        else:
            self._memory[self._position] = transition
            max_prio = self._priorities.max()
        self._priorities[self._position] = max_prio if max_prio > 0 else 1.0
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], List[int], np.ndarray]:
        """
        Sample a batch by priority (proportional to priority^alpha).
        Returns (transitions, indices, importance_sampling_weights).
        """
        n = len(self._memory)
        if n < batch_size:
            raise ValueError(f"Not enough transitions: {n} < {batch_size}")

        priorities = self._priorities[:n] + self.epsilon
        probs = np.power(priorities, self.alpha)
        probs /= probs.sum()
        indices = np.random.choice(n, size=batch_size, replace=True, p=probs)

        # Importance sampling weights: w_i = (1/(N*P(i)))^beta, normalized so max = 1
        N = n
        weights = np.power(1.0 / (N * probs[indices]), self.beta)
        weights /= weights.max()

        transitions = [self._memory[i] for i in indices]
        if self.beta_increment is not None and self.beta < self.beta_end:
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

        return transitions, indices.tolist(), weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities for sampled indices using |TD error|."""
        for idx, err in zip(indices, td_errors):
            self._priorities[idx] = float(np.abs(err)) + self.epsilon

    def __len__(self) -> int:
        return self._size

    def get_state(self) -> Tuple[List[Transition], np.ndarray, int, int, float]:
        """State for checkpointing: (transitions, priorities, position, size, beta)."""
        return (
            list(self._memory),
            self._priorities.copy(),
            self._position,
            self._size,
            self.beta,
        )

    def set_state(self, state: Tuple[List[Transition], np.ndarray, int, int, float]) -> None:
        """Restore from checkpoint."""
        self._memory = state[0][:]
        self._priorities = state[1].copy()
        self._position = state[2]
        self._size = state[3]
        self.beta = state[4]
