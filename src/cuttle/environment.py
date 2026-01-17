import random
import time
from typing import Optional

import gymnasium as gym
import numpy as np
from cuttle.actions import ActionRegistry


class CuttleEnvironment:

    # Initializes the environment and defines the observation and action spaces
    def __init__(self) -> None:
        """
        Initialize the Cuttle game environment.
        """

        # Generates the zones
        # A zone is a bool np array
        # Where a card is can be determined as follows: 13*suit + rank,
        # where suit is 0-3 and rank is 0-12
        self.dealer_hand = np.zeros(52, dtype=bool)
        self.dealer_field = np.zeros(52, dtype=bool)
        self.player_field = np.zeros(52, dtype=bool)
        self.player_hand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)

        #self.pain_lock = threading.Lock()

        # Special Zones which tell a player information about the game state
        # Stack: boolean array where stack[card_index] = True if that card is involved in the current stack
        self.stack = np.zeros(52, dtype=bool)
        # Internal tracking of stack action types for game logic (depth 0-4, values 0-53)
        self._stack_action_types = [0, 0, 0, 0, 0]
        self.dealer_revealed = np.zeros(
            52, dtype=bool
        )  # What cards in the dealer's hand that are public
        self.player_revealed = np.zeros(
            52, dtype=bool
        )  # What cards in the player's hand that are public
        # Effect-Shown: boolean array where effect_shown[card_index] = True if that card is shown by effect
        self.effect_shown = np.zeros(52, dtype=bool)
        self.top_deck = []
        self.current_bounced = []
        self.off_bounced = []
        # Revealed: When a card becomes known to the opponent, it becomes revealed
        # Defines who owns what zones, allows for easy access to fields
        self.player_zones = {
            "Hand": self.player_hand,
            "Field": self.player_field,
            "Revealed": self.player_revealed,
        }
        self.dealer_zones = {
            "Hand": self.dealer_hand,
            "Field": self.dealer_field,
            "Revealed": self.dealer_revealed,
        }

        # Swapped by passControl(), always start with the player_
        self.current_zones = self.player_zones
        self.off_zones = self.dealer_zones

        # Generates the cards for easy access to rank and suit based on index (demonstrated above)
        self.card_dict = self.generateCards()

        # For quick reference we get organize the indicies. The first split is royal and point (8s are in point), then the sublists are by rank
        self.point_indicies = []
        for rank in range(10):
            rank_list = []
            for suit in range(4):
                rank_list.append(self.getIndex(rank, suit))
            self.point_indicies.append(rank_list)

        self.royal_indicies = []
        for rank in range(10, 13):
            rank_list = []
            for suit in range(4):
                rank_list.append(self.getIndex(rank, suit))
            self.royal_indicies.append(rank_list)

        self.cur_eight_royals = [False, False, False, False]
        self.off_eight_royals = [False, False, False, False]

        self.cur_seen = []
        self.off_seen = []

        self.cur_queens = 0
        self.off_queens = 0

        self.jackreg = {}

        self.countered = False
        self.passed = False
        
        # Create action registry (generates static card structure internally)
        # CuttleEnvironment doesn't need card_dict/royal_indicies/point_indicies anymore
        # - scoreState() doesn't use them (iterates by index)
        # - updateRevealed() can check rank directly instead of using point_indicies[7]
        # - All action logic moves to action classes
        self.action_registry = ActionRegistry()
        
        # Keep card_dict and indices for backward compatibility during migration
        # These are now accessed via action_registry but kept here for existing code
        self.card_dict = self.action_registry.card_dict
        self.royal_indicies = self.action_registry.royal_indicies
        self.point_indicies = self.action_registry.point_indicies
        
        # Legacy action_to_move for backward compatibility (will be removed)
        self.action_to_move, self.actions_legacy = self.generateActions()
        self.actions = self.action_registry.total_actions

        self.one_offs = {self.aceAction: 1, self.twoAction: 2, self.threeAction: 3, self.fourAction: 4, self.fiveAction: 5, self.sixAction: 6, self.sevenAction01: 7, self.nineAction: 8, self.twoCounter:2}
        # Gym helps us out so we make gym spaces
        self.observation_space = self.get_obs()

        self.action_space = gym.spaces.Discrete(self.actions)

    def get_obs(self):
        # Slight abstraction here, the current_ zones are the current_ player_s field and hand,
        # while off_ zones are the opposite player_'s hand and field
        # This allows passControl to affect what will be visible to who
        # when turns or priority changes.
        self.updateRevealed()
        obs = {
            "Current Zones": self.current_zones,
            "Off-Player Field": self.off_zones["Field"],
            "Off-Player Revealed": self.off_zones["Revealed"],
            "Deck": self.deck,
            "Scrap": self.scrap,
            "Stack": self.stack,
            "Effect-Shown": self.effect_shown,
        }
        
        return obs

    def _get_info(self):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        random.seed(seed)
        # Reset to open state to make new game
        self.dealer_hand = np.zeros(52, dtype=bool)
        self.dealer_field = np.zeros(52, dtype=bool)
        self.player_field = np.zeros(52, dtype=bool)
        self.player_hand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)
        
        # Reset revealed zones
        self.dealer_revealed = np.zeros(52, dtype=bool)
        self.player_revealed = np.zeros(52, dtype=bool)
        
        # Reset game state flags
        self.countered = False
        self.passed = False
        self.stack = np.zeros(52, dtype=bool)
        self._stack_action_types = [0, 0, 0, 0, 0]
        self.effect_shown = np.zeros(52, dtype=bool)
        self.top_deck = []
        
        # Reset bounced cards
        self.current_bounced = []
        self.off_bounced = []
        
        # Reset eight royal tracking
        self.cur_eight_royals = [False, False, False, False]
        self.off_eight_royals = [False, False, False, False]
        
        # Reset seen cards tracking
        self.cur_seen = []
        self.off_seen = []
        
        # Reset queen counts
        self.cur_queens = 0
        self.off_queens = 0
        
        # Reset jack registry
        self.jackreg = {}

        # Makes sure all the zones are in the right places
        self.player_zones = {
            "Hand": self.player_hand,
            "Field": self.player_field,
            "Revealed": self.player_revealed,
        }
        self.dealer_zones = {
            "Hand": self.dealer_hand,
            "Field": self.dealer_field,
            "Revealed": self.dealer_revealed,
        }

        self.current_zones = self.player_zones
        self.off_zones = self.dealer_zones

        # Draw opening hands using new action system
        draw_action = self.action_registry.get_action(0)  # Draw action is always index 0
        if draw_action:
            self.passControl()
            for _ in range(6):
                draw_action.execute(self)
            
            self.passControl()
            for _ in range(5):
                draw_action.execute(self)

    # Converts an action into a move using the new action class system
    def step(self, action: int):
        # Use new action system
        action_obj = self.action_registry.get_action(action)
        
        if action_obj is None:
            # Fallback to legacy system for compatibility
            act = self.action_to_move.get(action)
            if act is None:
                return None, 0, False, True
            func = act[0]  # type: ignore
            args = act[1]  # type: ignore
            func(args)
        else:
            # Execute using new action class
            action_obj.execute(self)
        
        ob = self.get_obs()
        score, threshold = self.scoreState()
        terminated = score >= threshold
        truncated = not np.any(self.deck)

        return ob, score, terminated, truncated

    def render(self):
        curr_hand = self.current_zones["Hand"]
        curr_field = self.current_zones["Field"]
        index = 0
        zone_string = ""
        zone_string += "Current hand"
        for suit in range(4):
            for rank in range(13):
                if curr_hand[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)

        index = 0
        zone_string = ""
        zone_string += "Current field"
        for suit in range(4):
            for rank in range(13):
                if curr_field[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)

        off_hand = self.off_zones["Hand"]
        off_field = self.off_zones["Field"]
        index = 0
        zone_string = ""
        zone_string += "Off field"
        for suit in range(4):
            for rank in range(13):
                if off_field[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)

        index = 0
        zone_string = ""
        zone_string += "Off hand"
        for suit in range(4):
            for rank in range(13):
                if off_hand[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)

        index = 0
        zone_string = "Scrap: "
        for suit in range(4):
            for rank in range(13):
                if self.scrap[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)
        print(f"Curr Player Score: {self.scoreState()}")

    def drawAction(self, *args):
        hand = self.current_zones.get("Hand")
        possible_draws = np.where(self.deck)[0]
        if possible_draws.any():
            # trunk-ignore(bandit/B311)
            if self.top_deck:
                index = self.top_deck[0]
                self.top_deck = []
            else:
                index = possible_draws[random.randint(0, len(possible_draws) - 1)]

            hand[index] = True  # type: ignore
            self.deck[index] = False
        else:
            return 1

        return "Draw"

    def scoreAction(self, card):
        hand = self.current_zones.get("Hand")
        field = self.current_zones.get("Field")

        hand[card] = False  # type: ignore
        field[card] = True  # type: ignore

        return f"Scored f{card}"

    def scuttleAction(self, cardAndTarget):
        hand = self.current_zones.get("Hand")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        card = cardAndTarget[0]
        target = cardAndTarget[1]

        hand[card] = False  # type: ignore
        oppfield[target] = False  # type: ignore
        scrap[card] = True
        scrap[target] = True

        return f"Scuttled {target} with {card}"

    def aceAction(self, card):
        hand = self.current_zones.get("Hand")
        selfField = self.current_zones.get("Field")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            for rank_list in self.point_indicies:
                for card in rank_list:
                    if oppfield[card] or selfField[card]:  # type:ignore
                        oppfield[card] = False  # type:ignore
                        selfField[card] = False  # type:ignore
                        scrap[card] = True

    def twoAction(self, cardAndTarget):
        hand = self.current_zones.get("Hand")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        card = cardAndTarget[0]
        target = cardAndTarget[1]

        hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            oppfield[target] = False  # type: ignore
            scrap[target] = True

        return f"Scrapped {target} with {card}"

    def twoCounter(self, card):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False
        scrap[card] = True

    def passPriority(self, *args):
        self.passed = True
        return "Passed Priority"

    def threeAction(self, cardAndTarget):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        card = cardAndTarget[0]
        target = cardAndTarget[1]

        hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            hand[target] = True  # type: ignore
            scrap[target] = False

        # Mark target in revealed later

        return f"Recovered {target} with {card}"

    # TODO
    def fourAction(self, card):
        if not self.countered:
            self._stack_action_types[0] = 4
            self.stack[card] = True

        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True

    def resolveFour(self, targets):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap
        self._stack_action_types[0] = 0
        self.stack[:] = False
        if len(targets) > 0:
            t1 = targets[0]
            hand[t1] = False  # type: ignore
            scrap[t1] = True
        if len(targets) > 1:
            t2 = targets[1]
            hand[t2] = False  # type: ignore
            scrap[t2] = True

    def fiveAction(self, card):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            self.drawAction()
            self.drawAction()

    def sixAction(self, card):
        hand = self.current_zones.get("Hand")
        selfField = self.current_zones.get("Field")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            for rank_list in self.royal_indicies:
                for card in rank_list:
                    if oppfield[card] or selfField[card]:  # type:ignore
                        oppfield[card] = False  # type:ignore
                        selfField[card] = False  # type:ignore
                        scrap[card] = True

    # TODO
    def sevenAction01(self, card):
        # If a seven isn't already being resolved, reveal the top
        if not self.countered:
            self._stack_action_types[0] = 7
            self.stack[card] = True
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        self.reveal_two()
        if not np.any(self.effect_shown):
            self._stack_action_types[0] = 0
            self.stack[:] = False

    def sevenAction02(self, target):
        field = self.current_zones.get("Field")

        field[target] = True  # type: ignore
        # effect_shown is now boolean array - target is already the card index
        effect_indices = np.where(self.effect_shown)[0]
        if len(effect_indices) >= 2:
            # Find the other card (not target)
            other_indices = [i for i in effect_indices if i != target]
            if other_indices:
                to_top = other_indices[0]
            else:
                to_top = effect_indices[0] if len(effect_indices) > 0 else target
        else:
            to_top = target

        self.top_deck = [to_top]
        self.deck[to_top] = True
        self.stack[:] = False
        self._stack_action_types = [0, 0, 0, 0, 0]
        self.effect_shown[:] = False

    # TODO
    def eightRoyal(self, card):
        hand = self.current_zones["Hand"]
        field = self.current_zones["Field"]

        hand[card] = False
        field[card] = True
        i = self.point_indicies[7].index(card[0])
        self.cur_eight_royals[i] = True

    def nineAction(self, cardtargetself_hit):
        curr_hand = self.current_zones.get("Hand")
        curr_field = self.current_zones.get("Field")
        off_field = self.off_zones.get("Field")
        off_hand = self.off_zones.get("Hand")
        scrap = self.scrap

        card = cardtargetself_hit[0]
        target = cardtargetself_hit[1]
        self_hit = cardtargetself_hit[2]

        curr_hand[card] = False  # type: ignore
        scrap[card] = True
        if not self.countered:
            if self_hit:
                curr_field[target] = False  # type: ignore
                curr_hand[target] = True  # type: ignore
                self.current_bounced = [target]
                self.current_zones["Revealed"][target] = True
                self.cur_seen.append(target)
            else:
                off_field[target] = False  # type: ignore
                off_hand[target] = True  # type: ignore
                self.off_bounced = [target]
                self.off_zones["Revealed"][target] = True
                self.off_seen.append(target)

    def jackPlay(self, cardandtarget):
        hand = self.current_zones["Hand"]
        field = self.current_zones["Field"]

        card = cardandtarget[0]
        target = cardandtarget[1]

        hand[card] = False
        field[card] = True

        if target in self.jackreg:
            self.jackreg[target][1].append(card)
        else:
            if field is self.player_field:
                self.jackreg.update({target: ("dealer", [card])})
            else:
                self.jackreg.update({target: ("player", [card])})

    def generateActions(self):
        # Initializes storage mediums
        act_dict = {}
        actions = 0

        # Adds draw action
        act_dict.update({actions: (self.drawAction, "")})
        actions += 1

        # Adds score actions
        for x in range(52):
            if x not in self.royal_indicies[0]:
                act_dict.update({actions: (self.scoreAction, x)})
                actions += 1

        # Adds Scuttle actions
        for x in range(52):
            card_used = self.card_dict[x]  # type: ignore
            if not any(x in royal_list for royal_list in self.royal_indicies):
                for y in range(52):
                    target = self.card_dict[y]  # type: ignore
                    if not any(y in royal_list for royal_list in self.royal_indicies):
                        if target["rank"] < card_used["rank"] or (target["rank"] == card_used["rank"] and target["suit"] < card_used["suit"]):  # type: ignore
                            act_dict.update({actions: (self.scuttleAction, [x, y])})
                            actions += 1

        # Ace special action: boardwipe
        for x in self.point_indicies[0]:
            # 13 cards per rank, we are looking for rank 0 (Ace)
            act_dict.update({actions: (self.aceAction, [x])})
            actions += 1

        # Two: scrap target royal (Counters come later)
        for x in self.point_indicies[1]:
            for royal_list in self.royal_indicies:
                for target in royal_list:
                    act_dict.update({actions: (self.twoAction, [x, target])})
                    actions += 1

        for x in self.point_indicies[1]:
            act_dict.update({actions: (self.twoCounter, [x])})
            actions += 1

        act_dict.update({actions: (self.passPriority, [])})
        actions += 1
        # Three: Grab a card from scrap
        for x in self.point_indicies[2]:
            for target in range(52):
                act_dict.update({actions: (self.threeAction, [x, target])})
                actions += 1

        for x in self.point_indicies[3]:
            act_dict.update({actions: (self.fourAction, [x])})
            actions += 1

        fourTargets = []
        act_dict.update({actions: (self.resolveFour, fourTargets)})
        actions += 1
        for x in range(52):
            fourTargets = [x]
            act_dict.update({actions: (self.resolveFour, fourTargets)})
            actions += 1
            for y in range(52):
                if x < y:
                    fourTargets = [x, y]
                    act_dict.update({actions: (self.resolveFour, fourTargets)})
                    actions += 1

        for x in self.point_indicies[4]:
            # 13 cards per rank, we are looking for rank 4 (Five)
            act_dict.update({actions: (self.fiveAction, [x])})
            actions += 1

        for x in self.point_indicies[5]:
            # 13 cards per rank, we are looking for rank 5 (Six)
            act_dict.update({actions: (self.sixAction, [x])})
            actions += 1

        for x in self.point_indicies[6]:
            act_dict.update({actions: (self.sevenAction01, [x])})
            actions += 1

        for x in range(52):
            act_dict.update({actions: (self.sevenAction02, [x])})
            actions += 1

        for x in self.point_indicies[7]:
            act_dict.update({actions: (self.eightRoyal, [x])})
            actions += 1

        for x in self.point_indicies[8]:
            # 13 cards per rank, we are looking for rank 8 (Nine), one for each card
            for y in range(52):
                # Checks to make sure we aren't adding the unecessary self bounce to the pool.
                if y != x:
                    act_dict.update({actions: (self.nineAction, [x, y, True])})
                    actions += 1
                    act_dict.update({actions: (self.nineAction, [x, y, False])})
                    actions += 1

        for x in self.royal_indicies[0]:
                for y in range(52):
                    if x != y:
                        act_dict.update({actions: (self.jackPlay, [x, y])})
                        actions += 1
        return act_dict, actions

    def generateActionMask(self, countering = False):
        """Get valid action indices for current game state using new action system."""
        return self.action_registry.get_valid_actions(self, countering)

    # Cards are generated as follows:
    # Generate all cards, in order (Ace = 0, King = 12), in a suit, then increase the suit
    # Ex. 0, 13, 26, and 39 are aces. Any card index is 13 * suit + rank
    def generateCards(self):
        cards = {}
        index = 0
        for suit in range(4):
            for rank in range(13):
                c = {"rank": rank, "suit": suit}
                cards.update({index: c})
                index += 1

        return cards

    def scoreState(self) -> tuple[int, int]:
        field_scored = self.current_zones["Field"]
        index = 0
        score = 0
        king_count = 0
        self.cur_queens = 0
        self.jackmovement()
        for _ in range(4):
            for rank in range(13):
                if field_scored[index]:
                    if rank == 12:
                        king_count += 1
                    elif rank == 11:
                        self.cur_queens += 1
                    elif rank != 10:
                        score += rank + 1
                index += 1

        # Since each king decreases threshold differently
        # we need to use a switch
        match king_count:
            case 1:
                threshold = 14
            case 2:
                threshold = 10
            case 3:
                threshold = 5
            case 4:
                threshold = 0
            case _:
                threshold = 21
        return score, threshold

    def passControl(self):
        temp = self.current_bounced
        self.current_bounced = self.off_bounced
        self.off_bounced = temp

        temp = self.cur_eight_royals
        self.cur_eight_royals = self.off_eight_royals
        self.off_eight_royals = temp

        temp = self.cur_queens
        self.cur_queens = self.off_queens
        self.off_queens = temp

        if self.current_zones is self.player_zones:
            self.current_zones = self.dealer_zones
            self.off_zones = self.player_zones
            return

        if self.current_zones is self.dealer_zones:
            self.current_zones = self.player_zones
            self.off_zones = self.dealer_zones
            return

    def getIndex(self, rank, suit):
        return (13 * suit) + rank

    def reveal_two(self):
        possible_draws = np.where(self.deck)[0]
        if possible_draws.any():
            # trunk-ignore(bandit/B311)
            index1 = possible_draws[random.randint(0, len(possible_draws) - 1)]
            self.deck[index1] = False
            self.effect_shown[index1] = True
        possible_draws = np.where(self.deck)[0]
        if possible_draws.any():
            # trunk-ignore(bandit/B311)
            index2 = possible_draws[random.randint(0, len(possible_draws) - 1)]
            self.deck[index2] = False
            self.effect_shown[index2] = True

    def end_turn(self):
        self.current_bounced = []

    def updateStack(self, action, depth = 0):
        # Use new action system
        action_obj = self.action_registry.get_action(action)
        action_type_value = 53  # Default: not a one-off action
        
        if action_obj is None:
            # Fallback to legacy system
            moveType = self.action_to_move.get(action)
            if moveType:
                moveType = moveType[0]
                if moveType in self.one_offs.keys():
                    action_type_value = self.one_offs[moveType]
                else:
                    action_type_value = 53
            else:
                action_type_value = 53
        else:
            # Map action class to stack value
            from cuttle.actions import (AceAction, TwoAction, ThreeAction, FourAction, 
                               FiveAction, SixAction, SevenAction01, NineAction, TwoCounter)
            
            if isinstance(action_obj, AceAction):
                action_type_value = 1
            elif isinstance(action_obj, TwoAction) or isinstance(action_obj, TwoCounter):
                action_type_value = 2
            elif isinstance(action_obj, ThreeAction):
                action_type_value = 3
            elif isinstance(action_obj, FourAction):
                action_type_value = 4
            elif isinstance(action_obj, FiveAction):
                action_type_value = 5
            elif isinstance(action_obj, SixAction):
                action_type_value = 6
            elif isinstance(action_obj, SevenAction01):
                action_type_value = 7
            elif isinstance(action_obj, NineAction):
                action_type_value = 8
            else:
                action_type_value = 53  # Not a one-off action
        
        # Track action type internally for game logic
        self._stack_action_types[depth] = action_type_value
        
        # Track card in boolean array (if action has a card attribute)
        if action_obj is not None and hasattr(action_obj, 'card'):
            card = action_obj.card
            if isinstance(card, (int, np.integer)) and 0 <= card < 52:
                self.stack[card] = True

    def checkResponses(self):
        if self.passed:
            self.passed = False
            return False
        response = False
        if self._stack_action_types[0] != 53 and self._stack_action_types[0] != 0:
            inhand = np.where(self.off_zones["Hand"])[0]

            for x in self.point_indicies[1]:
                if x in inhand:
                    response = True
        return response

    def stackTop(self):
        return self._stack_action_types[0]

    def emptyStack(self):
        self.stack[:] = False
        self._stack_action_types = [0, 0, 0, 0, 0]

    def resolveStack(self):
        counters = 0
        for x in range(1,5):
            if self._stack_action_types[x] == 2:
                counters += 1
            self._stack_action_types[x] = 0
        if counters % 2 == 1:
            self._stack_action_types[0] = 0
            self.stack[:] = False
        self.countered = counters % 2 == 1

    def updateRevealed(self):

        for x in self.point_indicies[7]:
            if not self.current_zones["Field"][x]:
                i = self.point_indicies[7].index(x)
                self.cur_eight_royals[i] = False

        if any(self.cur_eight_royals):
            # Copy the hand to revealed (not assign reference!)
            np.copyto(self.off_zones["Revealed"], self.off_zones["Hand"])
        else:
            self.off_zones["Revealed"][:] = False

        for x in self.off_seen:
            if self.off_zones["Hand"][x]:
                self.off_zones["Revealed"][x] = True
            else:
                self.off_zones["Revealed"][x] = False

    def jackmovement(self):
        pfield = self.player_field
        dfield = self.dealer_field
        killed_keys = []
        for x in self.jackreg.keys():
            if pfield[x] or dfield[x]:
                pfield[x] = False
                dfield[x] = False

                if self.jackreg[x][0] == "player":
                    pfield[x] = True
                else:
                    dfield[x] = True

                for card in self.jackreg[x][1]:
                    if not (pfield[card] or dfield[card]):
                        self.jackreg[x][1].remove(card)

                if len(self.jackreg[x][1]) % 2 == 1:
                    if self.jackreg[x][0] == "player":
                        dfield[x] = True
                        pfield[x] = False
                    else:
                        pfield[x] = True
                        dfield[x] = False
            else:
                for card in self.jackreg[x][1]:
                    pfield[card] = False
                    dfield[card] = False
                    self.scrap[card] = True
                killed_keys.append(x)
        for key in killed_keys:
            self.jackreg.pop(key, 0)

