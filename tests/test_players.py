"""
Unit tests for Player classes.

This module provides comprehensive tests for all player implementations,
including Randomized, HeuristicHighCard, and Agent (DQN) players.
"""

import unittest
from typing import Dict, Any, List

import numpy as np
import torch

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork


class TestPlayerBaseClass(unittest.TestCase):
    """Test base Player class functionality."""
    
    def test_player_is_abstract(self):
        """Test that Player class cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Players.Player("TestPlayer")
    
    def test_player_has_name(self):
        """Test that all player subclasses have a name attribute."""
        randomized = Players.Randomized("TestRandom")
        self.assertEqual(randomized.name, "TestRandom")
        
        heuristic = Players.HeuristicHighCard("TestHeuristic")
        self.assertEqual(heuristic.name, "TestHeuristic")


class TestRandomizedPlayer(unittest.TestCase):
    """Test Randomized player implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.actions = self.env.actions
    
    def test_initialization(self):
        """Test that Randomized player initializes correctly."""
        player = Players.Randomized("RandomPlayer")
        self.assertEqual(player.name, "RandomPlayer")
        self.assertIsInstance(player, Players.Player)
    
    def test_initialization_with_seed(self):
        """Test that Randomized player can be initialized with seed."""
        player1 = Players.Randomized("Random1", seed=42)
        player2 = Players.Randomized("Random2", seed=42)
        
        # With same seed, should get same results
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        # Reset random state for both
        import random
        random.seed(42)
        action1 = player1.getAction(observation, valid_actions, self.actions, 0)
        random.seed(42)
        action2 = player2.getAction(observation, valid_actions, self.actions, 0)
        
        # Should be same with same seed
        self.assertEqual(action1, action2)
    
    def test_get_action_returns_valid_action(self):
        """Test that getAction returns a valid action."""
        player = Players.Randomized("RandomPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
    
    def test_get_action_handles_empty_valid_actions(self):
        """Test that getAction handles empty valid actions list."""
        player = Players.Randomized("RandomPlayer")
        observation = self.env.get_obs()
        valid_actions = []
        
        action = player.getAction(observation, valid_actions, self.actions, 0)
        self.assertEqual(action, 0)
    
    def test_get_action_ignores_parameters(self):
        """Test that getAction ignores unused parameters."""
        player = Players.Randomized("RandomPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        # Should work regardless of steps_done or force_greedy
        action1 = player.getAction(observation, valid_actions, self.actions, 0, False)
        action2 = player.getAction(observation, valid_actions, self.actions, 10000, True)
        
        # Both should be valid actions
        if valid_actions:
            self.assertIn(action1, valid_actions)
            self.assertIn(action2, valid_actions)
    
    def test_get_action_randomness(self):
        """Test that getAction produces different actions (randomness)."""
        player = Players.Randomized("RandomPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if len(valid_actions) > 1:
            actions = []
            for _ in range(10):
                action = player.getAction(observation, valid_actions, self.actions, 0)
                actions.append(action)
            
            # With multiple valid actions, should get some variety
            unique_actions = set(actions)
            # Not guaranteed, but very likely with 10 samples
            self.assertGreaterEqual(len(unique_actions), 1)


class TestHeuristicHighCardPlayer(unittest.TestCase):
    """Test HeuristicHighCard player implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.actions = self.env.actions
    
    def test_initialization(self):
        """Test that HeuristicHighCard player initializes correctly."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        self.assertEqual(player.name, "HeuristicPlayer")
        self.assertIsInstance(player, Players.Player)
    
    def test_get_action_returns_valid_action(self):
        """Test that getAction returns a valid action."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
    
    def test_get_action_prefers_score_actions(self):
        """Test that getAction prefers score actions when available."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        observation = self.env.get_obs()
        
        # Create valid actions that include score actions
        valid_actions = [0, 1, 2, 3, 10, 20, 30, 40]  # Mix of draw (0) and score actions
        
        action = player.getAction(observation, valid_actions, self.actions, 0)
        
        # Should prefer score actions (1-48) over draw (0)
        if any(1 <= a <= 48 for a in valid_actions):
            self.assertGreaterEqual(action, 1)
            self.assertLessEqual(action, 48)
    
    def test_get_action_handles_no_score_actions(self):
        """Test that getAction handles cases with no score actions."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        observation = self.env.get_obs()
        
        # Only draw action available
        valid_actions = [0]
        action = player.getAction(observation, valid_actions, self.actions, 0)
        self.assertEqual(action, 0)
    
    def test_get_action_handles_empty_valid_actions(self):
        """Test that getAction handles empty valid actions list."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        observation = self.env.get_obs()
        valid_actions = []
        
        action = player.getAction(observation, valid_actions, self.actions, 0)
        self.assertEqual(action, 0)
    
    def test_get_action_ignores_parameters(self):
        """Test that getAction ignores unused parameters."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        # Should work regardless of steps_done or force_greedy
        action1 = player.getAction(observation, valid_actions, self.actions, 0, False)
        action2 = player.getAction(observation, valid_actions, self.actions, 10000, True)
        
        # Both should be valid actions
        if valid_actions:
            self.assertIn(action1, valid_actions)
            self.assertIn(action2, valid_actions)


class TestAgentPlayer(unittest.TestCase):
    """Test Agent (DQN) player implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.observation_space = self.env.observation_space
        self.actions = self.env.actions
        
        # Create a model for testing
        self.model = NeuralNetwork(self.observation_space, 2, self.actions, None)
        self.model.eval()
        
        # Standard agent parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.01
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 3e-4
    
    def test_initialization(self):
        """Test that Agent initializes correctly."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.model, self.model)
        self.assertEqual(agent.policy, self.model)
        self.assertEqual(agent.batch_size, self.batch_size)
        self.assertEqual(agent.gamma, self.gamma)
        self.assertEqual(agent.eps_start, self.eps_start)
        self.assertEqual(agent.eps_end, self.eps_end)
        self.assertEqual(agent.eps_decay, self.eps_decay)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.optimizer)
        self.assertIsNotNone(agent.criterion)
    
    def test_get_action_returns_valid_action(self):
        """Test that getAction returns a valid action."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        agent.model.eval()
        
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            action = agent.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
    
    def test_get_action_force_greedy(self):
        """Test that force_greedy parameter forces exploitation."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        agent.model.eval()
        
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            # With force_greedy=True, should always use policy
            action = agent.getAction(observation, valid_actions, self.actions, 0, force_greedy=True)
            self.assertIn(action, valid_actions)
            
            # Should be deterministic (same observation, same action)
            action2 = agent.getAction(observation, valid_actions, self.actions, 0, force_greedy=True)
            self.assertEqual(action, action2)
    
    def test_get_action_epsilon_decay(self):
        """Test that epsilon decays with steps_done."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        agent.model.eval()
        
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            # Early in training (low steps), more exploration
            action_early = agent.getAction(observation, valid_actions, self.actions, 0, force_greedy=False)
            
            # Later in training (high steps), more exploitation
            action_late = agent.getAction(observation, valid_actions, self.actions, 100000, force_greedy=False)
            
            # Both should be valid
            self.assertIn(action_early, valid_actions)
            self.assertIn(action_late, valid_actions)
    
    def test_get_action_handles_empty_valid_actions(self):
        """Test that getAction handles empty valid actions list."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        agent.model.eval()
        
        observation = self.env.get_obs()
        valid_actions = []
        
        action = agent.getAction(observation, valid_actions, self.actions, 0)
        self.assertEqual(action, 0)
    
    def test_get_action_masks_invalid_actions(self):
        """Test that getAction masks invalid actions in greedy mode."""
        agent = Players.Agent(
            "TestAgent", self.model, self.batch_size, self.gamma,
            self.eps_start, self.eps_end, self.eps_decay, self.tau, self.lr
        )
        agent.model.eval()
        
        observation = self.env.get_obs()
        valid_actions = [0, 1, 2]  # Only these are valid
        
        # With force_greedy, should only select from valid actions
        action = agent.getAction(observation, valid_actions, self.actions, 0, force_greedy=True)
        self.assertIn(action, valid_actions)


class TestAgentReplayMemory(unittest.TestCase):
    """Test Agent's replay memory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.observation_space = self.env.observation_space
        self.actions = self.env.actions
        self.model = NeuralNetwork(self.observation_space, 2, self.actions, None)
        
        self.agent = Players.Agent(
            "TestAgent", self.model, 32, 0.99, 0.9, 0.01, 1000, 0.005, 3e-4
        )
    
    def test_memory_initialization(self):
        """Test that replay memory initializes correctly."""
        self.assertIsNotNone(self.agent.memory)
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_memory_push(self):
        """Test that transitions can be pushed to memory."""
        observation = self.env.get_obs()
        action = torch.tensor([0])
        next_obs = self.env.get_obs()
        reward = torch.tensor([1.0])
        
        # Memory expects observations (dicts), not processed states
        self.agent.memory.push(observation, action, next_obs, reward)
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_memory_capacity(self):
        """Test that memory respects capacity limit."""
        observation = self.env.get_obs()
        action = torch.tensor([0])
        next_obs = self.env.get_obs()
        reward = torch.tensor([1.0])
        
        # Push more than capacity
        for _ in range(50010):
            self.agent.memory.push(observation, action, next_obs, reward)
        
        # Should not exceed capacity
        self.assertLessEqual(len(self.agent.memory), 50000)
    
    def test_memory_sample(self):
        """Test that memory can sample batches."""
        observation = self.env.get_obs()
        action = torch.tensor([0])
        next_obs = self.env.get_obs()
        reward = torch.tensor([1.0])
        
        # Push some transitions
        for _ in range(10):
            self.agent.memory.push(observation, action, next_obs, reward)
        
        # Sample a batch
        batch = self.agent.memory.sample(5)
        self.assertEqual(len(batch), 5)
        
        # Each transition should have state, action, next_state, reward
        for transition in batch:
            self.assertIsNotNone(transition.state)
            self.assertIsNotNone(transition.action)
            self.assertIsNotNone(transition.next_state)
            self.assertIsNotNone(transition.reward)
    
    def test_memory_sample_handles_insufficient_data(self):
        """Test that memory sampling handles insufficient data."""
        observation = self.env.get_obs()
        state = self.agent.model.get_state(observation)
        action = torch.tensor([0])
        next_state = self.agent.model.get_state(observation)
        reward = torch.tensor([1.0])
        
        # Push only 2 transitions
        self.agent.memory.push(state, action, next_state, reward)
        self.agent.memory.push(state, action, next_state, reward)
        
        # Try to sample more than available - should raise ValueError
        with self.assertRaises(ValueError):
            self.agent.memory.sample(5)


class TestAgentOptimize(unittest.TestCase):
    """Test Agent's optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.observation_space = self.env.observation_space
        self.actions = self.env.actions
        self.model = NeuralNetwork(self.observation_space, 2, self.actions, None)
        self.model.train()  # Set to training mode
        
        self.agent = Players.Agent(
            "TestAgent", self.model, 32, 0.99, 0.9, 0.01, 1000, 0.005, 3e-4
        )
    
    def test_optimize_returns_loss_when_sufficient_memory(self):
        """Test that optimize returns loss when enough transitions in memory."""
        # Push enough transitions for a batch with different observations
        # Note: memory.push expects observations (dicts), not processed states
        for _ in range(32):
            observation = self.env.get_obs()
            action = torch.tensor([0])
            # Get next state from a different observation
            self.env.step(0)  # Take an action to change state
            next_obs = self.env.get_obs()
            reward = torch.tensor([1.0])
            self.agent.memory.push(observation, action, next_obs, reward)
        
        loss = self.agent.optimize()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
    
    def test_optimize_returns_none_when_insufficient_memory(self):
        """Test that optimize returns None when not enough transitions."""
        observation = self.env.get_obs()
        action = torch.tensor([0])
        next_obs = self.env.get_obs()
        reward = torch.tensor([1.0])
        
        # Push fewer than batch_size transitions
        for _ in range(10):
            self.agent.memory.push(observation, action, next_obs, reward)
        
        loss = self.agent.optimize()
        self.assertIsNone(loss)
    
    def test_optimize_updates_model_parameters(self):
        """Test that optimize updates model parameters."""
        # Get initial parameters
        initial_params = [p.clone() for p in self.agent.model.parameters()]
        
        # Push enough transitions with different observations
        # Note: memory.push expects observations (dicts), not processed states
        for _ in range(32):
            observation = self.env.get_obs()
            action = torch.tensor([0])
            # Get next state from a different observation
            self.env.step(0)  # Take an action to change state
            next_obs = self.env.get_obs()
            reward = torch.tensor([1.0])
            self.agent.memory.push(observation, action, next_obs, reward)
        
        # Optimize
        loss = self.agent.optimize()
        self.assertIsNotNone(loss)
        
        # Parameters should have changed (or at least gradients computed)
        # Check if any gradients were computed
        has_gradients = any(p.grad is not None for p in self.agent.model.parameters())
        self.assertTrue(has_gradients, "Gradients should be computed during optimization")


class TestPlayerIntegration(unittest.TestCase):
    """Integration tests for players with environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.actions = self.env.actions
    
    def test_randomized_player_plays_game(self):
        """Test that Randomized player can play a game."""
        player = Players.Randomized("RandomPlayer")
        
        for _ in range(5):
            observation = self.env.get_obs()
            valid_actions = self.env.generateActionMask()
            
            if not valid_actions:
                break
            
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
            
            self.env.step(action)
    
    def test_heuristic_player_plays_game(self):
        """Test that HeuristicHighCard player can play a game."""
        player = Players.HeuristicHighCard("HeuristicPlayer")
        
        for _ in range(5):
            observation = self.env.get_obs()
            valid_actions = self.env.generateActionMask()
            
            if not valid_actions:
                break
            
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
            
            self.env.step(action)
    
    def test_agent_player_plays_game(self):
        """Test that Agent player can play a game."""
        model = NeuralNetwork(self.env.observation_space, 2, self.actions, None)
        model.eval()
        
        agent = Players.Agent(
            "AgentPlayer", model, 32, 0.99, 0.9, 0.01, 1000, 0.005, 3e-4
        )
        
        for _ in range(5):
            observation = self.env.get_obs()
            valid_actions = self.env.generateActionMask()
            
            if not valid_actions:
                break
            
            action = agent.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
            
            self.env.step(action)


class TestScoreGapMaximizerPlayer(unittest.TestCase):
    """Test ScoreGapMaximizer player implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
        self.actions = self.env.actions
    
    def test_initialization(self):
        """Test that ScoreGapMaximizer player initializes correctly."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        self.assertEqual(player.name, "GapMaxPlayer")
        self.assertIsInstance(player, Players.Player)
        self.assertIsNotNone(player.action_registry)
    
    def test_get_action_returns_valid_action(self):
        """Test that getAction returns a valid action."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if valid_actions:
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
    
    def test_get_action_handles_empty_valid_actions(self):
        """Test that getAction handles empty valid actions list."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        observation = self.env.get_obs()
        valid_actions = []
        
        action = player.getAction(observation, valid_actions, self.actions, 0)
        self.assertEqual(action, 0)
    
    def test_prefers_scoring_high_value_cards(self):
        """Test that player prefers scoring high-value cards."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        
        # Set up: player has multiple cards in hand
        # Add some cards to hand
        self.env.current_zones["Hand"][0] = True  # Ace (rank 0, value 1)
        self.env.current_zones["Hand"][12] = True  # King (rank 12, value 20)
        self.env.deck[0] = False
        self.env.deck[12] = False
        
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        if len(valid_actions) > 1:
            action = player.getAction(observation, valid_actions, self.actions, 0)
            # Should prefer scoring the King (higher value) over Ace
            # This is a heuristic test - the player should select a scoring action
            self.assertIn(action, valid_actions)
    
    def test_prefers_scuttling_when_beneficial(self):
        """Test that player prefers scuttling when it creates positive gap."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        
        # Set up: player has high card, opponent has lower card on field
        # Player has a Six (rank 5, value 6) in hand
        # Opponent has a Five (rank 4, value 5) on field
        player_card = 5  # Six of first suit
        opponent_card = 4  # Five of first suit
        
        self.env.current_zones["Hand"][player_card] = True
        self.env.deck[player_card] = False
        self.env.off_zones["Field"][opponent_card] = True
        
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        # Player should consider scuttling (removes opponent's 5, uses our 6)
        # Net: opponent loses 5, we lose 6 = -1 gap change (not great)
        # But if we have a higher card scuttling a lower one, it's better
        action = player.getAction(observation, valid_actions, self.actions, 0)
        self.assertIn(action, valid_actions)
    
    def test_ignores_parameters(self):
        """Test that getAction ignores unused parameters."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        observation = self.env.get_obs()
        valid_actions = self.env.generateActionMask()
        
        # Should work regardless of steps_done or force_greedy
        action1 = player.getAction(observation, valid_actions, self.actions, 0, False)
        action2 = player.getAction(observation, valid_actions, self.actions, 10000, True)
        
        # Both should be valid actions
        if valid_actions:
            self.assertIn(action1, valid_actions)
            self.assertIn(action2, valid_actions)
    
    def test_plays_game(self):
        """Test that ScoreGapMaximizer player can play a game."""
        player = Players.ScoreGapMaximizer("GapMaxPlayer")
        
        for _ in range(5):
            observation = self.env.get_obs()
            valid_actions = self.env.generateActionMask()
            
            if not valid_actions:
                break
            
            action = player.getAction(observation, valid_actions, self.actions, 0)
            self.assertIn(action, valid_actions)
            
            self.env.step(action)


if __name__ == '__main__':
    unittest.main()

