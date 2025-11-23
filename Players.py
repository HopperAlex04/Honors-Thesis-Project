"""
Player classes for Cuttle game environment.
Provides base Player interface and concrete implementations for different strategies.
"""

import math
import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import List, Optional

import torch


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
        from GameEnvironment import CuttleEnvironment
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
        from Actions import ScoreAction, ScuttleAction, AceAction
        
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
            from Actions import (
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
        lr: float
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
        """
        super().__init__(name)
        
        # Set up policy model
        self.model = model
        self.policy = model

        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        # Replay Memory
        self.memory = ReplayMemory(50000)

        # Using Adam optimization
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=0
        )

        # Using Huber Loss (Smooth L1 Loss)
        self.criterion = torch.nn.SmoothL1Loss()

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
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * steps_done / self.eps_decay
        )

        # Use policy (exploit) if random sample exceeds threshold or forced greedy
        if sample > eps_threshold or force_greedy:
            with torch.no_grad():
                # Get Q-values for all actions from the policy network
                q_values = self.policy(observation)
                
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
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        # The zip function: zip(iterator1, iterator2...) returns a zip object, which is an iterator of the form [(iterator1[0], iterator2[0]...]),...]

        # zip(*iterable) will use all of the elements of that iterable as arguements. zip(*transitions) is the collection of all the transitions such that
        #   (assuming array structure) each row corresponds to all of that part of the transition. ex. zip(*transitions)[0] would be all of the states, in order.

        # Transition(*zip(*transitions)) is the final step, which makes this collection able to be accesed as so: batch.field, where field is one of the namedTuple's names
        # defined as Transition(transition, (state, action, next_state, reward))
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
        next_state_values = torch.zeros(self.batch_size)
        
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                # Get max Q-value for next states
                next_state_values[non_final_mask] = (
                    self.policy(non_final_next_states).max(1).values
                )
        
        # Compute expected Q-values using Bellman equation: Q(s,a) = r + γ * max Q(s',a')
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch  # Shape: [batch_size]
        expected_state_action_values = expected_state_action_values.unsqueeze(1)  # Shape: [batch_size, 1]

        # Compute Huber loss (Smooth L1 Loss)
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        
        self.optimizer.step()
        return loss.item()


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

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions from replay memory.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            List of Transition tuples
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.memory)
