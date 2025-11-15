import random
from typing import Optional

import gymnasium as gym
import numpy as np


class CuttleEnvironment:

    # Initializes the environment and defines the observation and action spaces
    def __init__(self) -> None:

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

        # Special Zones which tell a player information about the game state
        self.stack = [
            0,
            0,
            0,
            0,
            0,
        ]  # Contains the actions and counters after an action has been decided, in a model this is embedded when the observation is passed in
        self.dealer_revealed = np.zeros(
            52, dtype=bool
        )  # What cards in the dealer's hand that are public
        self.player_revealed = np.zeros(
            52, dtype=bool
        )  # What cards in the player's hand that are public
        self.effect_shown = [
            0,
            0,
        ]  # Contains the cards revealed by an effect after an action has been decided, in a model this is embedded when the observation is passed in
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

        # Generates the actions, as well as determining how many actions are in the environment.
        # Actions from the action_to_move dict are of the form (moveType, [args]),
        # where moveType is one of the functions below.
        self.action_to_move, self.actions = self.generateActions()

        # Gym helps us out so we make gym spaces
        self.observation_space = self.get_obs()

        self.action_space = gym.spaces.Discrete(self.actions)

    def get_obs(self):
        # Slight abstraction here, the current_ zones are the current_ player_s field and hand,
        # while off_ zones are the opposite player_'s hand and field
        # This allows passControl to affect what will be visible to who
        # when turns or priority changes.
        return {
            "Current Zones": self.current_zones,
            "Off-Player Field": self.off_zones["Field"],
            "Off-Player Revealed": self.off_zones["Revealed"],
            "Deck": self.deck,
            "Scrap": self.scrap,
            "Stack": self.stack,
            "Effect-Shown": self.effect_shown,
        }

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

        # Draw opening hands
        draw = self.action_to_move.get(0)
        args = draw[1]  # type: ignore
        draw = draw[0]  # type: ignore
        self.passControl()
        for _ in range(6):
            draw(args)

        self.passControl()
        for _ in range(5):
            draw(args)

    # Converts an action into a move by grabbing the calling the function with args from the move dict
    def step(self, action: int):
        act = self.action_to_move.get(action)

        # This is to prevent a crash in the event of exhausting all possible actions, for games this ends the game
        if act is None:
            return None, 0, False, True
        func = act[0]  # type: ignore
        args = act[1]  # type: ignore
        func(args)
        ob = self.get_obs()
        score, threshold = self.scoreState()
        terminated = score >= threshold
        truncated = False

        # ob is of the form [dict, dict] and should be broken up when reading a state
        # print(act)
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
        oppfield[target] = False  # type: ignore
        scrap[card] = True
        scrap[target] = True

        return f"Scrapped {target} with {card}"

    def threeAction(self, cardAndTarget):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        card = cardAndTarget[0]
        target = cardAndTarget[1]

        hand[card] = False  # type: ignore
        hand[target] = True  # type: ignore
        scrap[card] = True
        scrap[target] = False

        # Mark target in revealed later

        return f"Recovered {target} with {card}"

    # TODO
    def fourAction(self, card):
        self.stack[0] = 4

        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True

    def resolveFour(self, targets):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap
        self.stack[0] = 0
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

        self.drawAction()
        self.drawAction()

    def sixAction(self, card):
        hand = self.current_zones.get("Hand")
        selfField = self.current_zones.get("Field")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        for rank_list in self.royal_indicies:
            for card in rank_list:
                if oppfield[card] or selfField[card]:  # type:ignore
                    oppfield[card] = False  # type:ignore
                    selfField[card] = False  # type:ignore
                    scrap[card] = True

    # TODO
    def sevenAction01(self, card):
        # If a seven isn't already being resolved, reveal the top
        self.stack[0] = 7
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False  # type: ignore
        scrap[card] = True
        self.reveal_two()
        if self.effect_shown == [0, 0]:
            self.stack[0] = 0

    def sevenAction02(self, target):
        field = self.current_zones.get("Field")

        field[target] = True  # type: ignore
        to_top = self.effect_shown[1 - self.effect_shown.index(target)] - 1

        self.top_deck = [to_top]
        self.deck[to_top] = True
        self.stack = [0, 0, 0, 0, 0]
        self.effect_shown = [0, 0]

    # TODO
    def eightRoyal(self):
        pass

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

        if self_hit:
            curr_field[target] = False  # type: ignore
            curr_hand[target] = True  # type: ignore
            self.current_bounced = [target]
        else:
            off_field[target] = False  # type: ignore
            off_hand[target] = True  # type: ignore
            self.off_bounced = [target]

    def generateActions(self):
        # Initializes storage mediums
        act_dict = {}
        actions = 0

        # Adds draw action
        act_dict.update({actions: (self.drawAction, "")})
        actions += 1

        # Adds score actions
        for x in range(52):
            act_dict.update({actions: (self.scoreAction, x)})
            actions += 1

        # Adds Scuttle actions
        for x in range(52):
            card_used = self.card_dict[x]  # type: ignore
            for y in range(52):
                target = self.card_dict[y]  # type: ignore
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

        for x in self.point_indicies[8]:
            # 13 cards per rank, we are looking for rank 8 (Nine), one for each card
            for y in range(52):
                # Checks to make sure we aren't adding the unecessary self bounce to the pool.
                if y != x:
                    act_dict.update({actions: (self.nineAction, [x, y, True])})
                    actions += 1
                    act_dict.update({actions: (self.nineAction, [x, y, False])})
                    actions += 1

        return act_dict, actions

    def generateActionMask(self):
        inhand = np.where(self.current_zones["Hand"])
        self_field = np.where(self.current_zones["Field"])
        onfield = np.where(self.off_zones["Field"])
        scrap = np.where(self.scrap)[0]
        # Need this later for four
        # trunk-ignore(ruff/F841)
        opp_hand = np.where(self.off_zones["Hand"])[0]

        valid_actions = []

        for act_index, move in self.action_to_move.items():
            moveType = move[0]
            args = move[1]
            if moveType == self.drawAction and self.stack[0] == 0:
                valid_actions.append(act_index)
            elif moveType == self.scoreAction and self.stack[0] == 0:
                card = args
                if card in inhand[0] and card not in self.current_bounced:
                    valid_actions.append(act_index)
            elif moveType == self.scuttleAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    target = args[1]

                    cRank = self.card_dict[card]["rank"]
                    cSuit = self.card_dict[card]["suit"]

                    tRank = self.card_dict[target]["rank"]
                    tSuit = self.card_dict[card]["suit"]

                    if (
                        onfield[0].size > 0
                        and target in onfield[0]
                        and (cRank > tRank or (cRank == tRank and cSuit > tSuit))
                    ):
                        valid_actions.append(act_index)
            elif moveType == self.aceAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.fiveAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.nineAction and self.stack[0] == 0:
                card = move[1][0]
                if card in inhand[0]:
                    target = move[1][1]
                    selfhit = move[1][2]

                    if selfhit and self_field[0].size > 0 and target in self_field[0]:
                        valid_actions.append(act_index)
                    elif not selfhit and onfield[0].size > 0 and target in onfield[0]:
                        valid_actions.append(act_index)
            elif moveType == self.twoAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    target = args[1]
                    if target in onfield[0]:
                        valid_actions.append(act_index)
            elif moveType == self.threeAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    target = args[1]
                    if target in scrap:
                        valid_actions.append(act_index)
            elif moveType == self.sixAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.sevenAction01 and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.sevenAction02 and self.stack[0] == 7:
                target = args[0]
                if target != 0 and target in self.effect_shown:
                    valid_actions.append(act_index)
            elif moveType == self.fourAction and self.stack[0] == 0:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.resolveFour and self.stack[0] == 4:
                if len(args) == 0:
                    if len(inhand[0]) == 0:
                        valid_actions.append(act_index)
                elif len(args) == 1:
                    if len(inhand[0]) == 1:
                        t1 = args[0]
                        if t1 in inhand[0]:
                            valid_actions.append(act_index)
                elif len(args) >= 2:
                    if len(inhand[0]) >= 2:
                        t1 = args[0]
                        t2 = args[1]
                        if t1 in inhand[0] and t2 in inhand[0]:
                            valid_actions.append(act_index)

        return valid_actions

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
        for _ in range(4):
            for rank in range(13):
                if field_scored[index]:
                    if rank == 12:
                        king_count += 1
                    else:
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
            case _:
                threshold = 21
        return score, threshold

    def passControl(self):
        temp = self.current_bounced
        self.current_bounced = self.off_bounced
        self.off_bounced = temp
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
            self.effect_shown[0] = index1 + 1
        possible_draws = np.where(self.deck)[0]
        if possible_draws.any():
            # trunk-ignore(bandit/B311)
            index2 = possible_draws[random.randint(0, len(possible_draws) - 1)]
            self.deck[index2] = False
            self.effect_shown[1] = index2 + 1

    def end_turn(self):
        self.current_bounced = []
