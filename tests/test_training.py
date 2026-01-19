"""
Unit tests for Training module.

This module provides comprehensive tests for training functionality,
including update_replay_memory with score-based rewards.
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List

import torch

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork
from cuttle import training as Training


class TestUpdateReplayMemory(unittest.TestCase):
    """Test update_replay_memory function."""
    
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
    
    def test_update_replay_memory_terminal_reward_single_state(self):
        """Test that terminal reward is applied to single state."""
        states = [self.env.get_obs()]
        actions = [0]
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_WIN, next_state=None
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        self.assertEqual(transition.reward.item(), Training.REWARD_WIN)
        self.assertIsNone(transition.next_state)
    
    def test_update_replay_memory_terminal_reward_multiple_states(self):
        """Test that all states in final turn get terminal reward."""
        states = [
            self.env.get_obs(),
            self.env.get_obs(),
            self.env.get_obs()
        ]
        actions = [0, 1, 2]
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_WIN, next_state=None
        )
        
        self.assertEqual(len(self.agent.memory), 3)
        # All transitions should have terminal reward
        for i in range(3):
            transition = self.agent.memory.memory[i]
            self.assertEqual(transition.reward.item(), Training.REWARD_WIN)
            # Final state should have next_state=None
            if i == 2:
                self.assertIsNone(transition.next_state)
            else:
                # Other states should point to next state in sequence
                self.assertIsNotNone(transition.next_state)
    
    def test_update_replay_memory_intermediate_reward_no_score_change(self):
        """Test intermediate reward with no score change."""
        states = [self.env.get_obs()]
        actions = [0]
        next_state = self.env.get_obs()
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_INTERMEDIATE, 
            next_state=next_state, score_change=0
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        # Should be base intermediate reward (0.0) + (0 * 0.1) = 0.0
        self.assertEqual(transition.reward.item(), Training.REWARD_INTERMEDIATE)
        self.assertIsNotNone(transition.next_state)
    
    def test_update_replay_memory_intermediate_reward_positive_score_change(self):
        """Test intermediate reward with positive score change.
        
        With USE_INTERMEDIATE_REWARDS=True, score_change contributes to the reward.
        """
        states = [self.env.get_obs()]
        actions = [0]
        next_state = self.env.get_obs()
        score_change = 5  # Scored 5 points
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_INTERMEDIATE,
            next_state=next_state, score_change=score_change
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        # With USE_INTERMEDIATE_REWARDS=True, score_change is applied
        # Expected reward is REWARD_INTERMEDIATE (0.0) + (5 * 0.01) = 0.05
        expected_reward = Training.REWARD_INTERMEDIATE + (score_change * Training.SCORE_REWARD_SCALE)
        self.assertAlmostEqual(transition.reward.item(), expected_reward, places=5)
        self.assertIsNotNone(transition.next_state)
    
    def test_update_replay_memory_intermediate_reward_negative_score_change(self):
        """Test intermediate reward with negative score change (e.g., scuttled).
        
        With USE_INTERMEDIATE_REWARDS=True, negative score_change contributes negatively.
        """
        states = [self.env.get_obs()]
        actions = [0]
        next_state = self.env.get_obs()
        score_change = -3  # Lost 3 points (e.g., from scuttling)
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_INTERMEDIATE,
            next_state=next_state, score_change=score_change
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        # With USE_INTERMEDIATE_REWARDS=True, score_change is applied
        # Expected reward is REWARD_INTERMEDIATE (0.0) + (-3 * 0.01) = -0.03
        expected_reward = Training.REWARD_INTERMEDIATE + (score_change * Training.SCORE_REWARD_SCALE)
        self.assertAlmostEqual(transition.reward.item(), expected_reward, places=5)
        self.assertIsNotNone(transition.next_state)
    
    def test_update_replay_memory_intermediate_multiple_states_with_score_change(self):
        """Test intermediate reward with multiple states and score change.
        
        With USE_INTERMEDIATE_REWARDS=True, score_change contributes to the reward.
        """
        states = [
            self.env.get_obs(),
            self.env.get_obs(),
            self.env.get_obs()
        ]
        actions = [0, 1, 2]
        next_state = self.env.get_obs()
        score_change = 4  # Scored 4 points during this turn
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_INTERMEDIATE,
            next_state=next_state, score_change=score_change
        )
        
        self.assertEqual(len(self.agent.memory), 3)
        # With USE_INTERMEDIATE_REWARDS=True, score_change is applied
        # Expected reward is REWARD_INTERMEDIATE (0.0) + (4 * 0.01) = 0.04
        expected_reward = Training.REWARD_INTERMEDIATE + (score_change * Training.SCORE_REWARD_SCALE)
        
        # All states should get the same reward (score change applies to entire turn)
        for i in range(3):
            transition = self.agent.memory.memory[i]
            self.assertAlmostEqual(transition.reward.item(), expected_reward, places=5)
            # Last state should point to next_state, others point to next in sequence
            if i == 2:
                self.assertEqual(transition.next_state, next_state)
            else:
                self.assertEqual(transition.next_state, states[i + 1])
    
    def test_update_replay_memory_terminal_ignores_score_change(self):
        """Test that terminal rewards ignore score_change parameter."""
        states = [self.env.get_obs()]
        actions = [0]
        score_change = 10  # Large score change
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_WIN,
            next_state=None, score_change=score_change
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        # Terminal reward should be used, not score-based reward
        self.assertEqual(transition.reward.item(), Training.REWARD_WIN)
        self.assertIsNone(transition.next_state)
    
    def test_update_replay_memory_empty_states_list(self):
        """Test that empty states list is handled gracefully."""
        states = []
        actions = []
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_WIN
        )
        
        # Should not add anything to memory
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_update_replay_memory_loss_reward(self):
        """Test that loss reward is handled correctly."""
        states = [self.env.get_obs()]
        actions = [0]
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_LOSS, next_state=None
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        self.assertEqual(transition.reward.item(), Training.REWARD_LOSS)
        self.assertIsNone(transition.next_state)
    
    def test_update_replay_memory_draw_reward(self):
        """Test that draw reward is handled correctly."""
        states = [self.env.get_obs()]
        actions = [0]
        
        Training.update_replay_memory(
            self.agent, states, actions, Training.REWARD_DRAW, next_state=None
        )
        
        self.assertEqual(len(self.agent.memory), 1)
        transition = self.agent.memory.memory[0]
        self.assertEqual(transition.reward.item(), Training.REWARD_DRAW)
        self.assertIsNone(transition.next_state)
    
    def test_update_replay_memory_score_reward_scale_constant(self):
        """Test that SCORE_REWARD_SCALE constant is defined and has correct value."""
        self.assertIsNotNone(Training.SCORE_REWARD_SCALE)
        self.assertEqual(Training.SCORE_REWARD_SCALE, 0.01)  # Reduced from 0.1 to prevent Q-value explosion
        self.assertIsInstance(Training.SCORE_REWARD_SCALE, float)


class TestTrainingConstants(unittest.TestCase):
    """Test training module constants."""
    
    def test_reward_constants_exist(self):
        """Test that all reward constants are defined."""
        self.assertIsNotNone(Training.REWARD_WIN)
        self.assertIsNotNone(Training.REWARD_LOSS)
        self.assertIsNotNone(Training.REWARD_DRAW)
        self.assertIsNotNone(Training.REWARD_INTERMEDIATE)
        self.assertIsNotNone(Training.SCORE_REWARD_SCALE)
    
    def test_reward_constants_values(self):
        """Test that reward constants have expected values."""
        self.assertEqual(Training.REWARD_WIN, 1.0)
        self.assertEqual(Training.REWARD_LOSS, -1.0)
        self.assertEqual(Training.REWARD_DRAW, 0.0)
        self.assertEqual(Training.REWARD_INTERMEDIATE, 0.0)
        self.assertEqual(Training.SCORE_REWARD_SCALE, 0.01)  # Reduced from 0.1 to prevent Q-value explosion
    
    def test_reward_constants_types(self):
        """Test that reward constants have correct types."""
        self.assertIsInstance(Training.REWARD_WIN, float)
        self.assertIsInstance(Training.REWARD_LOSS, float)
        self.assertIsInstance(Training.REWARD_DRAW, float)
        self.assertIsInstance(Training.REWARD_INTERMEDIATE, float)
        self.assertIsInstance(Training.SCORE_REWARD_SCALE, float)


if __name__ == '__main__':
    unittest.main()

