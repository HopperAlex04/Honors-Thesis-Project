"""
Action classes for Cuttle game environment.
Implements action system with DQN-compatible integer indices.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import random
import numpy as np


class Action(ABC):
    """Base class for all game actions."""
    
    def __init__(self, action_id: int, args: Any = None):
        self.action_id = action_id  # The integer index for DQN
        self.args = args
    
    @abstractmethod
    def execute(self, game_state) -> Optional[str]:
        """Execute the action on the game state. Returns description or None."""
        pass
    
    @abstractmethod
    def validate(self, game_state, countering: bool = False) -> bool:
        """Check if this action is valid in the current game state."""
        pass
    
    def __eq__(self, other):
        """Actions are equal if they have same type and args."""
        return isinstance(other, type(self)) and self.args == other.args
    
    def __hash__(self):
        """Make actions hashable for use in sets/dicts."""
        return hash((type(self).__name__, self.args))
    
    def __repr__(self):
        return f"{type(self).__name__}(id={self.action_id}, args={self.args})"


class DrawAction(Action):
    """Action to draw a card from the deck."""
    
    MAX_HAND_SIZE = 8  # Maximum hand size (can draw if hand <= this value)
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        possible_draws = np.where(game_state.deck)[0]
        if possible_draws.any():
            if game_state.top_deck:
                index = game_state.top_deck[0]
                game_state.top_deck = []
            else:
                index = possible_draws[random.randint(0, len(possible_draws) - 1)]
            hand[index] = True
            game_state.deck[index] = False
            return "Draw"
        return None
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering:
            return False
        if game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return len(inhand) <= self.MAX_HAND_SIZE  # Can draw if hand size <= 8


class ScoreAction(Action):
    """Action to score a card (move from hand to field)."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        field = game_state.current_zones.get("Field")
        hand[self.card] = False
        field[self.card] = True
        return f"Scored {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        royal_jacks = game_state.action_registry.royal_indicies[0]
        return (self.card in inhand and 
                self.card not in game_state.current_bounced and
                self.card not in royal_jacks)


class ScuttleAction(Action):
    """Action to scuttle an opponent's card."""
    
    def __init__(self, action_id: int, card: int, target: int):
        super().__init__(action_id, (card, target))
        self.card = card
        self.target = target
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        oppfield = game_state.off_zones.get("Field")
        scrap = game_state.scrap
        
        hand[self.card] = False
        oppfield[self.target] = False
        scrap[self.card] = True
        scrap[self.target] = True
        return f"Scuttled {self.target} with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        onfield = np.where(game_state.off_zones["Field"])[0]
        
        if self.card not in inhand or self.target not in onfield:
            return False
        
        card_dict = game_state.action_registry.card_dict
        cRank = card_dict[self.card]["rank"]
        cSuit = card_dict[self.card]["suit"]
        tRank = card_dict[self.target]["rank"]
        tSuit = card_dict[self.target]["suit"]
        
        return (cRank > tRank or (cRank == tRank and cSuit > tSuit))


class AceAction(Action):
    """Action: Ace - Board wipe (destroys all point cards)."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        selfField = game_state.current_zones.get("Field")
        oppfield = game_state.off_zones.get("Field")
        scrap = game_state.scrap
        
        hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            point_indicies = game_state.action_registry.point_indicies
            for rank_list in point_indicies:
                for card_idx in rank_list:
                    if oppfield[card_idx] or selfField[card_idx]:
                        oppfield[card_idx] = False
                        selfField[card_idx] = False
                        scrap[card_idx] = True
        return f"Ace board wipe with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class TwoAction(Action):
    """Action: Two - Scrap target royal card."""
    
    def __init__(self, action_id: int, card: int, target: int):
        super().__init__(action_id, (card, target))
        self.card = card
        self.target = target
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        oppfield = game_state.off_zones.get("Field")
        scrap = game_state.scrap
        
        hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            oppfield[self.target] = False
            scrap[self.target] = True
        return f"Scrapped {self.target} with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        onfield = np.where(game_state.off_zones["Field"])[0]
        
        if self.card not in inhand or self.target not in onfield:
            return False
        
        # Queen protection check
        if game_state.off_queens == 1:
            royal_queens = game_state.action_registry.royal_indicies[1]
            if self.target in royal_queens:
                return False
        
        return True


class TwoCounter(Action):
    """Action: Two as counter - Cancel an opponent's one-off action."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        hand[self.card] = False
        scrap[self.card] = True
        return f"Countered with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if not countering:
            return False
        if game_state.off_queens != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class PassPriority(Action):
    """Action: Pass priority during counter phase."""
    
    def execute(self, game_state):
        game_state.passed = True
        return "Passed Priority"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        return countering


class ThreeAction(Action):
    """Action: Three - Recover a card from scrap."""
    
    def __init__(self, action_id: int, card: int, target: int):
        super().__init__(action_id, (card, target))
        self.card = card
        self.target = target
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        
        hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            hand[self.target] = True
            scrap[self.target] = False
        return f"Recovered {self.target} with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        scrap_cards = np.where(game_state.scrap)[0]
        return (self.card in inhand and self.target in scrap_cards)


class FourAction(Action):
    """Action: Four - Two-stage action to scrap cards from hand."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        if not game_state.countered:
            game_state._stack_action_types[0] = 4
            game_state.stack[self.card] = True
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        hand[self.card] = False
        scrap[self.card] = True
        return f"Played Four {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class ResolveFour(Action):
    """Action: Resolve Four - Choose cards to scrap."""
    
    def __init__(self, action_id: int, targets: list):
        super().__init__(action_id, targets)
        self.targets = targets
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        game_state._stack_action_types[0] = 0
        game_state.stack[:] = False
        if len(self.targets) > 0:
            t1 = self.targets[0]
            hand[t1] = False
            scrap[t1] = True
        if len(self.targets) > 1:
            t2 = self.targets[1]
            hand[t2] = False
            scrap[t2] = True
        return f"Resolved Four with targets {self.targets}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 4:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        if len(self.targets) == 0:
            return len(inhand) == 0
        elif len(self.targets) == 1:
            return len(inhand) == 1 and self.targets[0] in inhand
        elif len(self.targets) >= 2:
            return (len(inhand) >= 2 and 
                    self.targets[0] in inhand and 
                    self.targets[1] in inhand)
        return False


class FiveAction(Action):
    """Action: Five - Draw 2 cards."""
    
    MAX_HAND_LIMIT = 9  # Absolute maximum hand size (cannot exceed this)
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            # Draw twice, respecting hand limit (max 9 cards)
            draw_action = DrawAction(0)
            current_hand_size = np.where(hand)[0].size
            if current_hand_size < self.MAX_HAND_LIMIT:
                draw_action.execute(game_state)
                current_hand_size = np.where(hand)[0].size
            if current_hand_size < self.MAX_HAND_LIMIT:
                draw_action.execute(game_state)
        return f"Five drew 2 cards with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        if self.card not in inhand:
            return False
        # After playing Five: hand - 1 cards, then draw up to 2
        # Final hand size: hand - 1 + 2 = hand + 1
        # Require: hand + 1 <= MAX_HAND_LIMIT, i.e., hand <= MAX_HAND_LIMIT - 1
        return len(inhand) <= self.MAX_HAND_LIMIT - 1


class SixAction(Action):
    """Action: Six - Royal wipe (destroys all royal cards)."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones.get("Hand")
        selfField = game_state.current_zones.get("Field")
        oppfield = game_state.off_zones.get("Field")
        scrap = game_state.scrap
        
        hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            royal_indicies = game_state.action_registry.royal_indicies
            for rank_list in royal_indicies:
                for card_idx in rank_list:
                    if oppfield[card_idx] or selfField[card_idx]:
                        oppfield[card_idx] = False
                        selfField[card_idx] = False
                        scrap[card_idx] = True
        return f"Six royal wipe with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class SevenAction01(Action):
    """Action: Seven - Stage 1 (reveal top 2 cards)."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        if not game_state.countered:
            game_state._stack_action_types[0] = 7
            game_state.stack[self.card] = True
        hand = game_state.current_zones.get("Hand")
        scrap = game_state.scrap
        hand[self.card] = False
        scrap[self.card] = True
        # Call reveal_two method
        game_state.reveal_two()
        if not np.any(game_state.effect_shown):
            game_state._stack_action_types[0] = 0
            game_state.stack[:] = False
        return f"Seven revealed cards with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class SevenAction02(Action):
    """Action: Seven - Stage 2 (choose revealed card)."""
    
    def __init__(self, action_id: int, target: int):
        super().__init__(action_id, target)
        self.target = target  # This is the card index (0-51), not card_index + 1
    
    def execute(self, game_state):
        field = game_state.current_zones.get("Field")
        field[self.target] = True
        
        # Get the other card and put it on top of deck
        # effect_shown is now boolean array, so target is already the card index
        effect_indices = np.where(game_state.effect_shown)[0]
        if len(effect_indices) >= 2:
            # Find the other card (not target)
            other_indices = [i for i in effect_indices if i != self.target]
            if other_indices:
                to_top = other_indices[0]
                game_state.top_deck = [to_top]
                game_state.deck[to_top] = True
        
        game_state.stack[:] = False
        game_state._stack_action_types = [0, 0, 0, 0, 0]
        game_state.effect_shown[:] = False
        return f"Seven chose card {self.target}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 7:
            return False
        # effect_shown is now boolean array, so target is already the card index
        # Check if this target is one of the revealed cards
        return self.target < 52 and game_state.effect_shown[self.target]


class EightRoyal(Action):
    """Action: Eight - Score Eight (reveals opponent's hand)."""
    
    def __init__(self, action_id: int, card: int):
        super().__init__(action_id, card)
        self.card = card
    
    def execute(self, game_state):
        hand = game_state.current_zones["Hand"]
        field = game_state.current_zones["Field"]
        hand[self.card] = False
        field[self.card] = True
        
        # Find which suit this eight is (0-3)
        point_indicies = game_state.action_registry.point_indicies
        suit_index = point_indicies[7].index(self.card)
        game_state.cur_eight_royals[suit_index] = True
        return f"Scored Eight {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        return self.card in inhand


class NineAction(Action):
    """Action: Nine - Bounce a card (return to hand)."""
    
    MAX_HAND_LIMIT = 9  # Absolute maximum hand size (cannot exceed this)
    
    def __init__(self, action_id: int, card: int, target: int, self_hit: bool):
        super().__init__(action_id, (card, target, self_hit))
        self.card = card
        self.target = target
        self.self_hit = self_hit
    
    def execute(self, game_state):
        curr_hand = game_state.current_zones.get("Hand")
        curr_field = game_state.current_zones.get("Field")
        off_field = game_state.off_zones.get("Field")
        off_hand = game_state.off_zones.get("Hand")
        scrap = game_state.scrap
        
        curr_hand[self.card] = False
        scrap[self.card] = True
        if not game_state.countered:
            if self.self_hit:
                curr_field[self.target] = False
                curr_hand[self.target] = True
                game_state.current_bounced = [self.target]
                game_state.current_zones["Revealed"][self.target] = True
                game_state.cur_seen.append(self.target)
            else:
                off_field[self.target] = False
                off_hand[self.target] = True
                game_state.off_bounced = [self.target]
                game_state.off_zones["Revealed"][self.target] = True
                game_state.off_seen.append(self.target)
        return f"Nine bounced {self.target} with {self.card}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        self_field = np.where(game_state.current_zones["Field"])[0]
        onfield = np.where(game_state.off_zones["Field"])[0]
        
        if self.card not in inhand:
            return False
        
        if self.self_hit:
            if self_field.size > 0 and self.target in self_field:
                # Self-bounce: play Nine (-1), bounce card to own hand (+1) = net 0
                # Final hand size = current hand size, must not exceed limit
                if len(inhand) > self.MAX_HAND_LIMIT:
                    return False
                # Queen protection
                if game_state.cur_queens == 1:
                    royal_queens = game_state.action_registry.royal_indicies[1]
                    if self.target in royal_queens:
                        return False
                return True
        else:
            if onfield.size > 0 and self.target in onfield:
                # Opponent bounce: their hand increases by 1
                # Check opponent's hand limit before bouncing to their hand
                off_hand_size = np.where(game_state.off_zones["Hand"])[0].size
                if off_hand_size + 1 > self.MAX_HAND_LIMIT:
                    return False
                # Queen protection
                if game_state.off_queens == 1:
                    royal_queens = game_state.action_registry.royal_indicies[1]
                    if self.target in royal_queens:
                        return False
                return True
        return False


class JackPlay(Action):
    """Action: Jack - Play Jack to control opponent's card."""
    
    def __init__(self, action_id: int, card: int, target: int):
        super().__init__(action_id, (card, target))
        self.card = card
        self.target = target
    
    def execute(self, game_state):
        hand = game_state.current_zones["Hand"]
        field = game_state.current_zones["Field"]
        hand[self.card] = False
        field[self.card] = True
        
        if self.target in game_state.jackreg:
            game_state.jackreg[self.target][1].append(self.card)
        else:
            if field is game_state.player_field:
                game_state.jackreg.update({self.target: ("dealer", [self.card])})
            else:
                game_state.jackreg.update({self.target: ("player", [self.card])})
        return f"Jack {self.card} targeting {self.target}"
    
    def validate(self, game_state, countering: bool = False) -> bool:
        if countering or game_state.stackTop() != 0:
            return False
        inhand = np.where(game_state.current_zones["Hand"])[0]
        onfield = np.where(game_state.off_zones["Field"])[0]
        royal_indicies = game_state.action_registry.royal_indicies
        
        if self.card not in inhand:
            return False
        if self.target not in onfield:
            return False
        # Cannot target royal cards
        if any(self.target in r_list for r_list in royal_indicies):
            return False
        return True


class ActionRegistry:
    """Manages action instances and their integer indices."""
    
    def __init__(self):
        # Generate static card structure data (doesn't depend on game state)
        self.card_dict = self._generate_cards()
        self.royal_indicies = self._generate_royal_indices()
        self.point_indicies = self._generate_point_indices()
        
        self.actions: list[Action] = []
        self.action_index_map: dict[int, Action] = {}  # index -> Action
        self._generate_all_actions()
    
    def _generate_cards(self) -> dict:
        """Generate card dictionary: index -> {rank, suit}."""
        cards = {}
        index = 0
        for suit in range(4):
            for rank in range(13):
                cards[index] = {"rank": rank, "suit": suit}
                index += 1
        return cards
    
    def _generate_point_indices(self) -> list[list[int]]:
        """Generate point card indices organized by rank."""
        point_indicies = []
        for rank in range(10):  # Ranks 0-9 (Ace through 9)
            rank_list = []
            for suit in range(4):
                rank_list.append(13 * suit + rank)
            point_indicies.append(rank_list)
        return point_indicies
    
    def _generate_royal_indices(self) -> list[list[int]]:
        """Generate royal card indices organized by rank."""
        royal_indicies = []
        for rank in range(10, 13):  # Ranks 10-12 (Jack, Queen, King)
            rank_list = []
            for suit in range(4):
                rank_list.append(13 * suit + rank)
            royal_indicies.append(rank_list)
        return royal_indicies
    
    def _generate_all_actions(self):
        """Generate all possible action instances and assign indices."""
        action_id = 0
        
        # Draw action
        self.actions.append(DrawAction(action_id))
        self.action_index_map[action_id] = self.actions[-1]
        action_id += 1
        
        # Score actions
        for card in range(52):
            if card not in self.royal_indicies[0]:  # Can't score Jacks
                self.actions.append(ScoreAction(action_id, card))
                self.action_index_map[action_id] = self.actions[-1]
                action_id += 1
        
        # Scuttle actions
        for card in range(52):
            card_info = self.card_dict[card]
            if not any(card in royal_list for royal_list in self.royal_indicies):
                for target in range(52):
                    target_info = self.card_dict[target]
                    if not any(target in royal_list for royal_list in self.royal_indicies):
                        if (target_info["rank"] < card_info["rank"] or 
                            (target_info["rank"] == card_info["rank"] and 
                             target_info["suit"] < card_info["suit"])):
                            self.actions.append(ScuttleAction(action_id, card, target))
                            self.action_index_map[action_id] = self.actions[-1]
                            action_id += 1
        
        # Ace actions
        for card in self.point_indicies[0]:
            self.actions.append(AceAction(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Two actions (scrap royal)
        for card in self.point_indicies[1]:
            for royal_list in self.royal_indicies:
                for target in royal_list:
                    self.actions.append(TwoAction(action_id, card, target))
                    self.action_index_map[action_id] = self.actions[-1]
                    action_id += 1
        
        # Two counter actions
        for card in self.point_indicies[1]:
            self.actions.append(TwoCounter(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Pass priority
        self.actions.append(PassPriority(action_id))
        self.action_index_map[action_id] = self.actions[-1]
        action_id += 1
        
        # Three actions
        for card in self.point_indicies[2]:
            for target in range(52):
                self.actions.append(ThreeAction(action_id, card, target))
                self.action_index_map[action_id] = self.actions[-1]
                action_id += 1
        
        # Four actions
        for card in self.point_indicies[3]:
            self.actions.append(FourAction(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Resolve Four actions
        fourTargets = []
        self.actions.append(ResolveFour(action_id, fourTargets))
        self.action_index_map[action_id] = self.actions[-1]
        action_id += 1
        for x in range(52):
            fourTargets = [x]
            self.actions.append(ResolveFour(action_id, fourTargets))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
            for y in range(52):
                if x < y:
                    fourTargets = [x, y]
                    self.actions.append(ResolveFour(action_id, fourTargets))
                    self.action_index_map[action_id] = self.actions[-1]
                    action_id += 1
        
        # Five actions
        for card in self.point_indicies[4]:
            self.actions.append(FiveAction(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Six actions
        for card in self.point_indicies[5]:
            self.actions.append(SixAction(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Seven stage 1 actions
        for card in self.point_indicies[6]:
            self.actions.append(SevenAction01(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Seven stage 2 actions
        for target in range(52):
            self.actions.append(SevenAction02(action_id, target))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Eight actions
        for card in self.point_indicies[7]:
            self.actions.append(EightRoyal(action_id, card))
            self.action_index_map[action_id] = self.actions[-1]
            action_id += 1
        
        # Nine actions
        for card in self.point_indicies[8]:
            for target in range(52):
                if target != card:
                    self.actions.append(NineAction(action_id, card, target, True))
                    self.action_index_map[action_id] = self.actions[-1]
                    action_id += 1
                    self.actions.append(NineAction(action_id, card, target, False))
                    self.action_index_map[action_id] = self.actions[-1]
                    action_id += 1
        
        # Jack actions
        for card in self.royal_indicies[0]:
            for target in range(52):
                if target != card:
                    self.actions.append(JackPlay(action_id, card, target))
                    self.action_index_map[action_id] = self.actions[-1]
                    action_id += 1
    
    def get_action(self, action_index: int) -> Optional[Action]:
        """Get action by integer index."""
        return self.action_index_map.get(action_index)
    
    def get_valid_actions(self, game_state, countering: bool = False) -> list[int]:
        """Get list of valid action indices for the current game state."""
        valid_indices = []
        for action_id, action in self.action_index_map.items():
            if action.validate(game_state, countering):
                valid_indices.append(action_id)
        return valid_indices
    
    @property
    def total_actions(self) -> int:
        """Total number of possible actions."""
        return len(self.actions)

