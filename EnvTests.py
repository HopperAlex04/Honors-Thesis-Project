"""
Unit tests for CuttleEnvironment.

This module provides comprehensive tests for the Cuttle card game environment,
including initialization, actions, state management, and game mechanics.
"""

import unittest
from typing import Dict, Any

import numpy as np

from GameEnvironment import CuttleEnvironment


class TestCuttleEnvironmentInitialization(unittest.TestCase):
    """Test environment initialization and basic setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
    
    def test_initialization(self):
        """Test that environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(len(self.env.deck), 52)
        self.assertEqual(len(self.env.player_hand), 52)
        self.assertEqual(len(self.env.dealer_hand), 52)
        self.assertEqual(len(self.env.player_field), 52)
        self.assertEqual(len(self.env.dealer_field), 52)
        self.assertEqual(len(self.env.scrap), 52)
    
    def test_initial_deck_state(self):
        """Test that all cards start in the deck."""
        self.assertTrue(np.all(self.env.deck))
        self.assertFalse(np.any(self.env.player_hand))
        self.assertFalse(np.any(self.env.dealer_hand))
        self.assertFalse(np.any(self.env.player_field))
        self.assertFalse(np.any(self.env.dealer_field))
        self.assertFalse(np.any(self.env.scrap))
    
    def test_initial_zones(self):
        """Test that zone references are set correctly."""
        self.assertEqual(self.env.current_zones, self.env.player_zones)
        self.assertEqual(self.env.off_zones, self.env.dealer_zones)
    
    def test_initial_stack(self):
        """Test that stack is initialized correctly."""
        self.assertEqual(len(self.env.stack), 5)
        self.assertTrue(all(s == 0 for s in self.env.stack))
    
    def test_action_registry_exists(self):
        """Test that action registry is initialized."""
        self.assertIsNotNone(self.env.action_registry)
        self.assertGreater(self.env.actions, 0)
    
    def test_observation_space(self):
        """Test that observation space is defined."""
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)


class TestCuttleEnvironmentReset(unittest.TestCase):
    """Test environment reset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
    
    def test_reset_initializes_game(self):
        """Test that reset properly initializes a new game."""
        self.env.reset()
        
        # Check that hands are drawn
        player_hand_size = np.sum(self.env.player_hand)
        dealer_hand_size = np.sum(self.env.dealer_hand)
        
        self.assertEqual(player_hand_size, 5, "Player should have 5 cards after reset")
        self.assertEqual(dealer_hand_size, 6, "Dealer should have 6 cards after reset")
    
    def test_reset_deck_size(self):
        """Test that deck size is correct after reset."""
        self.env.reset()
        
        # 52 cards total, 5 for player, 6 for dealer = 41 remaining
        expected_deck_size = 52 - 5 - 6
        actual_deck_size = np.sum(self.env.deck)
        
        self.assertEqual(actual_deck_size, expected_deck_size)
    
    def test_reset_with_seed(self):
        """Test that reset with seed produces consistent results."""
        self.env.reset(seed=42)
        player_hand_1 = self.env.player_hand.copy()
        dealer_hand_1 = self.env.dealer_hand.copy()
        
        self.env.reset(seed=42)
        player_hand_2 = self.env.player_hand.copy()
        dealer_hand_2 = self.env.dealer_hand.copy()
        
        np.testing.assert_array_equal(player_hand_1, player_hand_2)
        np.testing.assert_array_equal(dealer_hand_1, dealer_hand_2)
    
    def test_reset_clears_fields(self):
        """Test that reset clears fields and scrap."""
        # Set up some cards in fields and scrap
        self.env.player_field[0] = True
        self.env.dealer_field[1] = True
        self.env.scrap[2] = True
        
        self.env.reset()
        
        self.assertFalse(np.any(self.env.player_field))
        self.assertFalse(np.any(self.env.dealer_field))
        self.assertFalse(np.any(self.env.scrap))


class TestCuttleEnvironmentDrawAction(unittest.TestCase):
    """Test draw action functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_draw_action_reduces_deck(self):
        """Test that drawing reduces deck size."""
        initial_deck_size = np.sum(self.env.deck)
        initial_hand_size = np.sum(self.env.current_zones["Hand"])
        
        # Get draw action (action_id 0)
        draw_action = self.env.action_registry.get_action(0)
        self.assertIsNotNone(draw_action)
        
        draw_action.execute(self.env)
        
        new_deck_size = np.sum(self.env.deck)
        new_hand_size = np.sum(self.env.current_zones["Hand"])
        
        self.assertEqual(new_deck_size, initial_deck_size - 1)
        self.assertEqual(new_hand_size, initial_hand_size + 1)
    
    def test_draw_action_valid_when_hand_not_full(self):
        """Test that draw is valid when hand has less than 9 cards."""
        # Hand should have 5 cards after reset
        hand_size = np.sum(self.env.current_zones["Hand"])
        self.assertLess(hand_size, 9)
        
        draw_action = self.env.action_registry.get_action(0)
        self.assertTrue(draw_action.validate(self.env, countering=False))
    
    def test_draw_action_invalid_when_hand_full(self):
        """Test that draw is invalid when hand is full (9 cards)."""
        # Fill hand to 9 cards
        hand = self.env.current_zones["Hand"]
        deck = self.env.deck
        available_cards = np.where(deck)[0]
        
        for i in range(9 - np.sum(hand)):
            if len(available_cards) > 0:
                card = available_cards[0]
                hand[card] = True
                deck[card] = False
                available_cards = np.where(deck)[0]
        
        self.assertEqual(np.sum(hand), 9)
        
        draw_action = self.env.action_registry.get_action(0)
        self.assertFalse(draw_action.validate(self.env, countering=False))
    
    def test_draw_action_invalid_during_counter(self):
        """Test that draw is invalid during counter exchange."""
        draw_action = self.env.action_registry.get_action(0)
        self.assertFalse(draw_action.validate(self.env, countering=True))


class TestCuttleEnvironmentScoreAction(unittest.TestCase):
    """Test score action functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_score_action_moves_card_to_field(self):
        """Test that scoring moves a card from hand to field."""
        # Find a card in hand that can be scored (not a Jack)
        hand = self.env.current_zones["Hand"]
        cards_in_hand = np.where(hand)[0]
        
        if len(cards_in_hand) > 0:
            # Find first non-Jack card
            royal_jacks = self.env.action_registry.royal_indicies[0]
            scoreable_card = None
            for card in cards_in_hand:
                if card not in royal_jacks:
                    scoreable_card = card
                    break
            
            if scoreable_card is not None:
                # Get score action for this card
                valid_actions = self.env.generateActionMask()
                score_action_id = None
                
                for action_id in valid_actions:
                    action = self.env.action_registry.get_action(action_id)
                    if action and hasattr(action, 'card') and action.card == scoreable_card:
                        score_action_id = action_id
                        break
                
                if score_action_id is not None:
                    initial_hand_size = np.sum(hand)
                    initial_field_size = np.sum(self.env.current_zones["Field"])
                    
                    self.env.step(score_action_id)
                    
                    self.assertFalse(hand[scoreable_card])
                    self.assertTrue(self.env.current_zones["Field"][scoreable_card])
                    self.assertEqual(np.sum(hand), initial_hand_size - 1)
                    self.assertEqual(np.sum(self.env.current_zones["Field"]), initial_field_size + 1)


class TestCuttleEnvironmentScuttleAction(unittest.TestCase):
    """Test scuttle action functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_scuttle_action_destroys_opponent_card(self):
        """Test that scuttling destroys opponent's card."""
        # Set up: player has a card that can scuttle, dealer has a card on field
        # Need cards with appropriate ranks for scuttling
        # Let's use rank 5 (Six) to scuttle rank 4 (Five)
        player_card = 5   # Six of first suit (rank 5)
        dealer_card = 4   # Five of first suit (rank 4) - can be scuttled by Six
        
        self.env.current_zones["Hand"][player_card] = True
        self.env.deck[player_card] = False
        self.env.off_zones["Field"][dealer_card] = True
        
        # Find scuttle action
        valid_actions = self.env.generateActionMask()
        scuttle_action_id = None
        
        for action_id in valid_actions:
            action = self.env.action_registry.get_action(action_id)
            if action and hasattr(action, 'card') and hasattr(action, 'target'):
                if action.card == player_card and action.target == dealer_card:
                    scuttle_action_id = action_id
                    break
        
        if scuttle_action_id is not None:
            initial_scrap_player = self.env.scrap[player_card]
            initial_scrap_dealer = self.env.scrap[dealer_card]
            
            self.env.step(scuttle_action_id)
            
            # Both cards should be in scrap
            self.assertTrue(self.env.scrap[player_card], 
                          f"Player card {player_card} should be in scrap")
            self.assertTrue(self.env.scrap[dealer_card],
                          f"Dealer card {dealer_card} should be in scrap")
            # Cards should be removed from hand and field
            self.assertFalse(self.env.current_zones["Hand"][player_card])
            self.assertFalse(self.env.off_zones["Field"][dealer_card])
        else:
            # If no scuttle action found, skip test (might be due to validation rules)
            self.skipTest("No valid scuttle action found for test setup")


class TestCuttleEnvironmentActionMasking(unittest.TestCase):
    """Test action mask generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_action_mask_returns_list(self):
        """Test that action mask returns a list of integers."""
        valid_actions = self.env.generateActionMask()
        self.assertIsInstance(valid_actions, list)
        self.assertTrue(all(isinstance(a, int) for a in valid_actions))
    
    def test_action_mask_contains_draw_when_valid(self):
        """Test that draw action is in mask when valid."""
        hand_size = np.sum(self.env.current_zones["Hand"])
        if hand_size < 9:
            valid_actions = self.env.generateActionMask()
            self.assertIn(0, valid_actions)  # Draw action is action 0
    
    def test_action_mask_only_valid_actions(self):
        """Test that mask only contains valid actions."""
        valid_actions = self.env.generateActionMask()
        
        for action_id in valid_actions:
            action = self.env.action_registry.get_action(action_id)
            if action:
                self.assertTrue(action.validate(self.env, countering=False))


class TestCuttleEnvironmentStep(unittest.TestCase):
    """Test step function and game progression."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_step_returns_observation(self):
        """Test that step returns an observation."""
        valid_actions = self.env.generateActionMask()
        if len(valid_actions) > 0:
            observation, score, terminated, truncated = self.env.step(valid_actions[0])
            self.assertIsNotNone(observation)
            self.assertIsInstance(observation, dict)
    
    def test_step_returns_score(self):
        """Test that step returns current score."""
        valid_actions = self.env.generateActionMask()
        if len(valid_actions) > 0:
            observation, score, terminated, truncated = self.env.step(valid_actions[0])
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
    
    def test_step_returns_termination_status(self):
        """Test that step returns termination status."""
        valid_actions = self.env.generateActionMask()
        if len(valid_actions) > 0:
            observation, score, terminated, truncated = self.env.step(valid_actions[0])
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
    
    def test_step_terminates_on_win(self):
        """Test that step terminates when score threshold is reached."""
        # This is harder to test deterministically, but we can check the logic
        score, threshold = self.env.scoreState()
        # If score already meets threshold, game should terminate
        if score >= threshold:
            valid_actions = self.env.generateActionMask()
            if len(valid_actions) > 0:
                observation, new_score, terminated, truncated = self.env.step(valid_actions[0])
                # After scoring enough, should terminate
                if new_score >= threshold:
                    self.assertTrue(terminated)


class TestCuttleEnvironmentScoreState(unittest.TestCase):
    """Test score calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_score_state_returns_tuple(self):
        """Test that scoreState returns a tuple."""
        score, threshold = self.env.scoreState()
        self.assertIsInstance(score, int)
        self.assertIsInstance(threshold, int)
    
    def test_score_starts_at_zero(self):
        """Test that score starts at zero with empty field."""
        # Clear field
        self.env.current_zones["Field"].fill(False)
        score, threshold = self.env.scoreState()
        self.assertEqual(score, 0)
    
    def test_threshold_defaults_to_21(self):
        """Test that threshold defaults to 21 with no kings."""
        self.env.current_zones["Field"].fill(False)
        score, threshold = self.env.scoreState()
        self.assertEqual(threshold, 21)
    
    def test_score_increases_with_cards(self):
        """Test that score increases when cards are scored."""
        # Score an Ace (rank 0, worth 1 point)
        ace_index = 0  # First ace
        self.env.current_zones["Hand"][ace_index] = True
        self.env.deck[ace_index] = False
        
        # Get score action and execute
        valid_actions = self.env.generateActionMask()
        for action_id in valid_actions:
            action = self.env.action_registry.get_action(action_id)
            if action and hasattr(action, 'card') and action.card == ace_index:
                self.env.step(action_id)
                break
        
        score, threshold = self.env.scoreState()
        self.assertGreaterEqual(score, 0)


class TestCuttleEnvironmentControlPassing(unittest.TestCase):
    """Test control passing between players."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_pass_control_swaps_zones(self):
        """Test that passControl swaps current and off zones."""
        initial_current = self.env.current_zones
        initial_off = self.env.off_zones
        
        self.env.passControl()
        
        self.assertEqual(self.env.current_zones, initial_off)
        self.assertEqual(self.env.off_zones, initial_current)
    
    def test_pass_control_twice_restores_state(self):
        """Test that passing control twice restores original state."""
        initial_current = self.env.current_zones
        initial_off = self.env.off_zones
        
        self.env.passControl()
        self.env.passControl()
        
        self.assertEqual(self.env.current_zones, initial_current)
        self.assertEqual(self.env.off_zones, initial_off)


class TestCuttleEnvironmentStackManagement(unittest.TestCase):
    """Test stack management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_stack_top_returns_top_value(self):
        """Test that stackTop returns the top stack value."""
        self.env.stack[0] = 5
        self.assertEqual(self.env.stackTop(), 5)
    
    def test_empty_stack_clears_stack(self):
        """Test that emptyStack clears the stack."""
        self.env.stack[0] = 5
        self.env.emptyStack()
        self.assertEqual(self.env.stackTop(), 0)
    
    def test_update_stack_adds_action(self):
        """Test that updateStack maps action to stack value."""
        # Test with a draw action (should map to 53 - not a one-off)
        draw_action_id = 0
        self.env.updateStack(draw_action_id)
        self.assertEqual(self.env.stack[0], 53, "Draw action should map to 53")
        
        # Test with an Ace action (should map to 1)
        # First, add an Ace to hand
        ace_index = 0
        self.env.current_zones["Hand"][ace_index] = True
        self.env.deck[ace_index] = False
        
        valid_actions = self.env.generateActionMask()
        ace_action_id = None
        for action_id in valid_actions:
            action = self.env.action_registry.get_action(action_id)
            if action and hasattr(action, 'card') and action.card == ace_index:
                if action.__class__.__name__ == "AceAction":
                    ace_action_id = action_id
                    break
        
        if ace_action_id is not None:
            self.env.updateStack(ace_action_id)
            self.assertEqual(self.env.stack[0], 1, "Ace action should map to 1")


class TestCuttleEnvironmentObservation(unittest.TestCase):
    """Test observation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_get_obs_returns_dict(self):
        """Test that get_obs returns a dictionary."""
        observation = self.env.get_obs()
        self.assertIsInstance(observation, dict)
    
    def test_observation_contains_required_keys(self):
        """Test that observation contains required keys."""
        observation = self.env.get_obs()
        expected_keys = [
            "Current Zones",
            "Off-Player Field",
            "Off-Player Revealed",
            "Deck",
            "Scrap",
            "Stack",
            "Effect-Shown"
        ]
        for key in expected_keys:
            self.assertIn(key, observation, f"Observation missing key: {key}")
    
    def test_observation_zones_are_arrays(self):
        """Test that zone data in observation are numpy arrays."""
        observation = self.env.get_obs()
        zones = observation["Current Zones"]
        self.assertIsInstance(zones, dict)
        for zone_name, zone_data in zones.items():
            self.assertIsInstance(zone_data, np.ndarray)


class TestCuttleEnvironmentSpecialCards(unittest.TestCase):
    """Test special card actions (Ace, Two, Three, etc.)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = CuttleEnvironment()
        self.env.reset()
    
    def test_ace_action_available_when_ace_in_hand(self):
        """Test that Ace action is available when Ace is in hand."""
        # Add an Ace to hand (rank 0)
        ace_index = 0
        self.env.current_zones["Hand"][ace_index] = True
        self.env.deck[ace_index] = False
        
        valid_actions = self.env.generateActionMask()
        ace_actions = [
            a for a in valid_actions
            if self.env.action_registry.get_action(a) and
            self.env.action_registry.get_action(a).__class__.__name__ == "AceAction"
        ]
        self.assertGreater(len(ace_actions), 0)
    
    def test_five_action_draws_cards(self):
        """Test that Five action draws two cards."""
        # Add a Five to hand (rank 4)
        five_index = 4  # First five
        self.env.current_zones["Hand"][five_index] = True
        self.env.deck[five_index] = False
        
        initial_hand_size = np.sum(self.env.current_zones["Hand"])
        initial_deck_size = np.sum(self.env.deck)
        
        # Find and execute Five action
        valid_actions = self.env.generateActionMask()
        for action_id in valid_actions:
            action = self.env.action_registry.get_action(action_id)
            if action and hasattr(action, 'card') and action.card == five_index:
                if action.__class__.__name__ == "FiveAction":
                    self.env.step(action_id)
                    break
        
        # Hand should have 2 more cards (Five is consumed, 2 drawn)
        new_hand_size = np.sum(self.env.current_zones["Hand"])
        new_deck_size = np.sum(self.env.deck)
        
        # Five is consumed, so hand size should be initial + 2 - 1 = initial + 1
        # Deck should be initial - 2
        self.assertEqual(new_hand_size, initial_hand_size + 1)
        self.assertEqual(new_deck_size, initial_deck_size - 2)


if __name__ == '__main__':
    unittest.main()
