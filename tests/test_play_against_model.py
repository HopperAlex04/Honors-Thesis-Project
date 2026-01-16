"""
Unit tests for play_against_model.py script.

Tests all functions and classes to prevent errors during gameplay.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import torch
import numpy as np

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the module under test
# We need to import it carefully since it has side effects
import play_against_model as play_module
from cuttle import players as Players
from cuttle.environment import CuttleEnvironment
from cuttle.networks import NeuralNetwork


class TestCardToString(unittest.TestCase):
    """Test card_to_string function."""
    
    def test_card_to_string_ace_of_spades(self):
        """Test converting card index 0 (Ace of Spades)."""
        result = play_module.card_to_string(0)
        self.assertEqual(result, "Ace of Spades")
    
    def test_card_to_string_king_of_clubs(self):
        """Test converting card index 51 (King of Clubs)."""
        result = play_module.card_to_string(51)
        self.assertEqual(result, "King of Clubs")
    
    def test_card_to_string_middle_card(self):
        """Test converting a middle card."""
        result = play_module.card_to_string(26)  # Ace of Diamonds
        self.assertEqual(result, "Ace of Diamonds")
    
    def test_card_to_string_all_suits(self):
        """Test cards from all suits."""
        # Ace of each suit
        self.assertEqual(play_module.card_to_string(0), "Ace of Spades")
        self.assertEqual(play_module.card_to_string(13), "Ace of Hearts")
        self.assertEqual(play_module.card_to_string(26), "Ace of Diamonds")
        self.assertEqual(play_module.card_to_string(39), "Ace of Clubs")
    
    def test_card_to_string_all_ranks(self):
        """Test all ranks in a suit."""
        suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
        ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        
        for suit_idx, suit in enumerate(suits):
            for rank_idx, rank in enumerate(ranks):
                card_idx = suit_idx * 13 + rank_idx
                expected = f"{rank} of {suit}"
                result = play_module.card_to_string(card_idx)
                self.assertEqual(result, expected, f"Failed for card index {card_idx}")


class TestActionToString(unittest.TestCase):
    """Test action_to_string function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_action_to_string_draw_action(self):
        """Test DrawAction conversion."""
        # Find a draw action
        valid_actions = self.env.generateActionMask()
        draw_actions = [a for a in valid_actions if self.env.action_registry.get_action(a).__class__.__name__ == "DrawAction"]
        if draw_actions:
            result = play_module.action_to_string(self.env, draw_actions[0])
            self.assertIn("Draw", result)
    
    def test_action_to_string_score_action(self):
        """Test ScoreAction conversion."""
        # Find a score action
        valid_actions = self.env.generateActionMask()
        for action_id in valid_actions:
            action_obj = self.env.action_registry.get_action(action_id)
            if action_obj and action_obj.__class__.__name__ == "ScoreAction":
                result = play_module.action_to_string(self.env, action_id)
                self.assertIn("Score", result)
                break
    
    def test_action_to_string_invalid_action(self):
        """Test handling of invalid action ID."""
        # Use a very large action ID that doesn't exist
        result = play_module.action_to_string(self.env, 99999)
        self.assertIn("Unknown", result)
    
    def test_action_to_string_scuttle_action(self):
        """Test ScuttleAction conversion."""
        valid_actions = self.env.generateActionMask()
        for action_id in valid_actions:
            action_obj = self.env.action_registry.get_action(action_id)
            if action_obj and action_obj.__class__.__name__ == "ScuttleAction":
                result = play_module.action_to_string(self.env, action_id)
                self.assertIn("Scuttle", result)
                break


class TestDisplayGameState(unittest.TestCase):
    """Test display_game_state function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_display_game_state_human_p1(self):
        """Test displaying game state when human is P1."""
        # Ensure environment has proper observation structure
        self.env.reset()
        obs = self.env.get_obs()
        
        # Check that required keys exist (they might be named differently)
        # If keys don't exist, the function will fail, which is what we're testing
        try:
            fake_out = StringIO()
            with patch('sys.stdout', fake_out):
                play_module.display_game_state(self.env, human_is_p1=True)
                output = fake_out.getvalue()
                
                # Check that output contains expected elements
                self.assertIn("GAME STATE", output)
                self.assertIn("You (P1)", output)
                self.assertIn("AI (P2)", output)
                self.assertIn("Scores", output)
        except KeyError as e:
            # If there's a KeyError, that's a bug we're catching
            self.fail(f"display_game_state raised KeyError: {e}. Observation keys: {list(obs.keys())}")
    
    def test_display_game_state_human_p2(self):
        """Test displaying game state when human is P2."""
        self.env.reset()
        obs = self.env.get_obs()
        
        try:
            fake_out = StringIO()
            with patch('sys.stdout', fake_out):
                play_module.display_game_state(self.env, human_is_p1=False)
                output = fake_out.getvalue()
                
                self.assertIn("GAME STATE", output)
                self.assertIn("You (P2)", output)
                self.assertIn("AI (P1)", output)
        except KeyError as e:
            self.fail(f"display_game_state raised KeyError: {e}. Observation keys: {list(obs.keys())}")
    
    def test_display_game_state_handles_empty_hand(self):
        """Test that display handles empty hand gracefully."""
        # Create environment with no cards in hand
        env = CuttleEnvironment()
        env.reset()
        # Manually clear hand (for testing)
        env.player_hand.fill(False)
        
        try:
            fake_out = StringIO()
            with patch('sys.stdout', fake_out):
                play_module.display_game_state(env, human_is_p1=True)
                output = fake_out.getvalue()
                # Check that it doesn't crash and produces some output
                self.assertGreater(len(output), 0)
        except KeyError as e:
            obs = env.get_obs()
            self.fail(f"display_game_state raised KeyError: {e}. Observation keys: {list(obs.keys())}")


class TestLoadModel(unittest.TestCase):
    """Test load_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_path = self.temp_dir / "test_checkpoint.pt"
        
        # Create a minimal valid checkpoint
        env = CuttleEnvironment()
        model = NeuralNetwork(env.observation_space, 16, env.actions, None)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'target_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_model_valid_checkpoint(self):
        """Test loading a valid checkpoint."""
        agent, env = play_module.load_model(self.checkpoint_path)
        
        self.assertIsInstance(agent, Players.Agent)
        self.assertIsInstance(env, CuttleEnvironment)
        self.assertEqual(agent.name, "AI")
    
    def test_load_model_no_features_detection(self):
        """Test that no_features checkpoints disable features."""
        # Create checkpoint with no_features in name
        no_features_path = self.temp_dir / "no_features_checkpoint0.pt"
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        torch.save(checkpoint, no_features_path)
        
        agent, env = play_module.load_model(no_features_path)
        
        self.assertFalse(env.include_highest_point_value)
        self.assertFalse(env.include_highest_point_value_opponent_field)
    
    def test_load_model_with_config(self):
        """Test loading model with custom config path."""
        # Create a config file with same embedding size as checkpoint
        # (checkpoint was created with embedding_size=16)
        config_path = self.temp_dir / "config.json"
        config = {
            "embedding_size": 16,  # Must match checkpoint
            "batch_size": 64,
            "gamma": 0.95,
            "eps_start": 0.95,
            "eps_end": 0.01,
            "eps_decay": 30000,
            "tau": 0.01,
            "target_update_frequency": 1000,
            "learning_rate": 1e-4,
            "replay_buffer_size": 50000,
        }
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        agent, env = play_module.load_model(self.checkpoint_path, config_path)
        
        # Check that agent was created with config values
        self.assertIsInstance(agent, Players.Agent)
        self.assertEqual(agent.batch_size, 64)
    
    def test_load_model_missing_config(self):
        """Test loading model when config file doesn't exist."""
        # Use a non-existent config path
        fake_config = self.temp_dir / "nonexistent.json"
        agent, env = play_module.load_model(self.checkpoint_path, fake_config)
        
        # Should still work with defaults
        self.assertIsInstance(agent, Players.Agent)
    
    def test_load_model_invalid_checkpoint(self):
        """Test loading an invalid checkpoint file."""
        invalid_path = self.temp_dir / "invalid.pt"
        invalid_path.write_text("not a valid checkpoint")
        
        with self.assertRaises(Exception):
            play_module.load_model(invalid_path)
    
    def test_load_model_nonexistent_checkpoint(self):
        """Test loading a non-existent checkpoint."""
        nonexistent = self.temp_dir / "nonexistent.pt"
        
        with self.assertRaises(FileNotFoundError):
            play_module.load_model(nonexistent)


class TestListCheckpoints(unittest.TestCase):
    """Test list_checkpoints function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_list_checkpoints_empty_directory(self):
        """Test listing checkpoints in empty directory."""
        checkpoints = play_module.list_checkpoints(self.temp_dir)
        self.assertEqual(len(checkpoints), 0)
    
    def test_list_checkpoints_with_files(self):
        """Test listing checkpoints when files exist."""
        # Create some checkpoint files
        (self.temp_dir / "checkpoint0.pt").touch()
        (self.temp_dir / "checkpoint1.pt").touch()
        (self.temp_dir / "other_file.txt").touch()  # Should be ignored
        
        checkpoints = play_module.list_checkpoints(self.temp_dir)
        
        self.assertEqual(len(checkpoints), 2)
        self.assertTrue(all(p.suffix == ".pt" for p in checkpoints))
    
    def test_list_checkpoints_sorted(self):
        """Test that checkpoints are returned sorted."""
        (self.temp_dir / "checkpoint2.pt").touch()
        (self.temp_dir / "checkpoint1.pt").touch()
        (self.temp_dir / "checkpoint0.pt").touch()
        
        checkpoints = play_module.list_checkpoints(self.temp_dir)
        
        # Check they're sorted
        names = [p.name for p in checkpoints]
        self.assertEqual(names, sorted(names))


class TestHumanPlayer(unittest.TestCase):
    """Test HumanPlayer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_human_player_initialization(self):
        """Test HumanPlayer initialization."""
        player = play_module.HumanPlayer("TestHuman", self.env)
        self.assertEqual(player.name, "TestHuman")
        self.assertEqual(player.env, self.env)
    
    @patch('play_against_model.get_human_action')
    def test_human_player_get_action(self, mock_get_action):
        """Test HumanPlayer.getAction calls get_human_action."""
        mock_get_action.return_value = 42
        player = play_module.HumanPlayer("TestHuman", self.env)
        
        valid_actions = [1, 2, 3]
        result = player.getAction({}, valid_actions, 100, 0, False)
        
        self.assertEqual(result, 42)
        mock_get_action.assert_called_once_with(self.env, valid_actions)


class TestGetHumanAction(unittest.TestCase):
    """Test get_human_action function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    @patch('builtins.input', return_value='0')
    @patch('sys.stdout', new=StringIO())
    def test_get_human_action_valid_input(self, mock_input):
        """Test getting action with valid input."""
        valid_actions = [10, 20, 30]
        result = play_module.get_human_action(self.env, valid_actions)
        
        self.assertEqual(result, 10)  # First action
    
    @patch('builtins.input', side_effect=['-1', '0'])
    @patch('sys.stdout', new=StringIO())
    def test_get_human_action_invalid_then_valid(self, mock_input):
        """Test getting action after invalid input."""
        valid_actions = [10, 20, 30]
        result = play_module.get_human_action(self.env, valid_actions)
        
        self.assertEqual(result, 10)
        self.assertEqual(mock_input.call_count, 2)
    
    @patch('builtins.input', side_effect=['abc', '1'])
    @patch('sys.stdout', new=StringIO())
    def test_get_human_action_non_numeric_then_valid(self, mock_input):
        """Test handling non-numeric input."""
        valid_actions = [10, 20, 30]
        result = play_module.get_human_action(self.env, valid_actions)
        
        self.assertEqual(result, 20)  # Second action
    
    @patch('builtins.input', return_value='')
    @patch('sys.stdout', new=StringIO())
    def test_get_human_action_empty_input(self, mock_input):
        """Test handling empty input."""
        valid_actions = [10, 20, 30]
        # Should keep asking, so we'll provide a valid input after
        with patch('builtins.input', side_effect=['', '0']):
            result = play_module.get_human_action(self.env, valid_actions)
            self.assertEqual(result, 10)
    
    @patch('builtins.input', side_effect=KeyboardInterrupt())
    def test_get_human_action_keyboard_interrupt(self, mock_input):
        """Test handling KeyboardInterrupt."""
        valid_actions = [10, 20, 30]
        with self.assertRaises(SystemExit):
            play_module.get_human_action(self.env, valid_actions)


class TestSelectCheckpoint(unittest.TestCase):
    """Test select_checkpoint function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('builtins.input', return_value='0')
    def test_select_checkpoint_valid_selection(self, mock_input):
        """Test selecting a checkpoint with valid input."""
        (self.temp_dir / "checkpoint0.pt").touch()
        (self.temp_dir / "checkpoint1.pt").touch()
        
        result = play_module.select_checkpoint(self.temp_dir)
        
        self.assertEqual(result.name, "checkpoint0.pt")
    
    @patch('builtins.input', side_effect=['invalid', '1'])
    def test_select_checkpoint_invalid_then_valid(self, mock_input):
        """Test selecting after invalid input."""
        (self.temp_dir / "checkpoint0.pt").touch()
        (self.temp_dir / "checkpoint1.pt").touch()
        
        result = play_module.select_checkpoint(self.temp_dir)
        
        self.assertEqual(result.name, "checkpoint1.pt")
    
    def test_select_checkpoint_no_checkpoints(self):
        """Test selecting when no checkpoints exist."""
        with self.assertRaises(SystemExit):
            play_module.select_checkpoint(self.temp_dir)


class TestIntegration(unittest.TestCase):
    """Integration tests for the play module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_path = self.temp_dir / "test_checkpoint.pt"
        
        # Create a valid checkpoint
        env = CuttleEnvironment()
        model = NeuralNetwork(env.observation_space, 16, env.actions, None)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'target_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_and_display_integration(self):
        """Test loading model and displaying game state."""
        agent, env = play_module.load_model(self.checkpoint_path)
        env.reset()
        
        try:
            fake_out = StringIO()
            with patch('sys.stdout', fake_out):
                play_module.display_game_state(env, human_is_p1=True)
                output = fake_out.getvalue()
                
                self.assertIn("GAME STATE", output)
                self.assertIsInstance(agent, Players.Agent)
        except KeyError as e:
            obs = env.get_obs()
            self.fail(f"display_game_state raised KeyError: {e}. Observation keys: {list(obs.keys())}")
    
    def test_action_conversion_integration(self):
        """Test action conversion with real environment."""
        agent, env = play_module.load_model(self.checkpoint_path)
        env.reset()
        
        valid_actions = env.generateActionMask()
        if valid_actions:
            action_str = play_module.action_to_string(env, valid_actions[0])
            self.assertIsInstance(action_str, str)
            self.assertGreater(len(action_str), 0)


if __name__ == '__main__':
    unittest.main()

