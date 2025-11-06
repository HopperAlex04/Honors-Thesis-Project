import random
from typing import Optional
import numpy as np
import gymnasium as gym



class CuttleEnvironment(gym.Env):

    #Initializes the environment and defines the observation and action spaces
    def __init__(self) -> None:

        #Generates the zones
        #A zone is a bool np array
        #Where a card is can be determined as follows: 13*suit + rank,
        # where suit is 0-3 and rank is 0-12
        self.dealer_hand = np.zeros(52, dtype=bool)
        self.dealer_field = np.zeros(52, dtype=bool)
        self.player_field = np.zeros(52, dtype=bool)
        self.player_hand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)

        #Defines who owns what zones, allows for easy access to fields
        self.player_zones = {"Hand": self.player_hand, "Field": self.player_field}
        self.dealer_zones = {"Hand": self.dealer_hand, "Field": self.dealer_field}

        #Swapped by passControl(), always start with the player_
        self.current_zones = self.player_zones
        self.off_zones = self.dealer_zones

        #Generates the cards for easy access to rank and suit based on index (demonstrated above)
        self.card_dict = self.generateCards()

        #Generates the actions, as well as determining how many actions are in the environment.
        #Actions from the action_to_move dict are of the form (moveType, [args]),
        # where moveType is one of the functions below.
        self.action_to_move, self.actions = self.generateActions()

        #Gym helps us out so we make gym spaces
        self.observation_space = gym.spaces.MultiBinary([6,52])
        self.action_space = gym.spaces.Discrete(self.actions)

    def get_obs(self):
        #Slight abstraction here, the current_ zones are the current_ player_s field and hand,
        # while off_ zones are the opposite player_'s hand and field
        #This allows passControl to affect what will be visible to who
        # when turns or priority changes.
        return {
            "Current Zones": self.current_zones,
            "Off-Player Zones": self.off_zones,
            "Deck": self.deck,
            "Scrap": self.scrap}

    def _get_info(self):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        #Reset to open state to make new game
        self.dealer_hand = np.zeros(52, dtype=bool)
        self.dealer_field = np.zeros(52, dtype=bool)
        self.player_field = np.zeros(52, dtype=bool)
        self.player_hand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)

        #Makes sure all the zones are in the right places
        self.player_zones = {"Hand": self.player_hand, "Field": self.player_field}
        self.dealer_zones = {"Hand": self.dealer_hand, "Field": self.dealer_field}

        self.current_zones = self.player_zones
        self.off_zones = self.dealer_zones

        #Draw opening hands
        draw = self.action_to_move.get(0)
        args = draw[1] # type: ignore
        draw = draw[0] # type: ignore
        self.passControl()
        for _ in range(6):
            draw(args)

        self.passControl()
        for _ in range(5):
            draw(args)

    #Converts an action into a move by grabbing the calling the function with args from the move dict
    def step(self, action:int):
        act = self.action_to_move.get(action)

        #This is to prevent a crash in the event of exhausting all possible actions, for games this ends the game
        if act is None:
            return None, 0, False, True
        func = act[0] # type: ignore
        args = act[1] # type: ignore
        func(args)
        ob = self.get_obs()
        score = self.scoreState()
        terminated = score >= 21
        truncated = False

        #ob is of the form [dict, dict] and should be broken up when reading a state
        return ob, score, terminated, truncated

    def render(self):
        currhand = self.current_zones["Hand"]
        currfield = self.current_zones["Field"]
        index = 0
        zone_string = ""
        zone_string += "Current hand"
        for suit in range(4):
            for rank in range(13):
                if currhand[index]:
                    zone_string += f" |{rank} {suit}| "
                index += 1
        print(zone_string)

        index = 0
        zone_string = ""
        zone_string += "Current field"
        for suit in range(4):
            for rank in range(13):
                if currfield[index]:
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
            index = possible_draws[random.randint(0, len(possible_draws) - 1)]
            hand[index] = True # type: ignore
            self.deck[index] = False
        else: return 1

        return "Draw"

    def scoreAction(self, card):
        hand = self.current_zones.get("Hand")
        field = self.current_zones.get("Field")


        hand[card] = False # type: ignore
        field[card] = True # type: ignore

        return f"Scored f{card}"

    def scuttleAction(self, cardAndTarget):
        hand = self.current_zones.get("Hand")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap


        card = cardAndTarget[0]
        target = cardAndTarget[1]

        hand[card] = False # type: ignore
        oppfield[target] = False # type: ignore
        scrap[card] = True
        scrap[target] = True

        return f"Scuttled {target} with {card}"

    def aceAction(self, card):
        hand = self.current_zones.get("Hand")
        oppfield = self.off_zones.get("Field")
        scrap = self.scrap

        hand[card] = False # type: ignore
        scrap[card] = True
        for x in range(oppfield.size): # type: ignore
            oppfield[x] = False # type: ignore
            scrap[x] = True

    def fiveAction(self, card):
        hand = self.current_zones.get("Hand")
        scrap = self.scrap

        hand[card] = False # type: ignore
        scrap[card] = True

        self.drawAction()
        self.drawAction()



    def generateActions(self):
        #Initializes storage mediums
        act_dict = {}
        actions = 0

        #Adds draw action
        act_dict.update({actions: (self.drawAction, "")})
        actions += 1

        #Adds score actions
        for x in range(52):
            act_dict.update({actions: (self.scoreAction, x)})
            actions += 1

        #Adds Scuttle actions
        for x in range(52):
            card_used = self.card_dict[x] # type: ignore
            for y in range(52):
                target = self.card_dict[y] # type: ignore
                if target["rank"] < card_used["rank"] or (target["rank"] == card_used["rank"] and target["suit"] < card_used["suit"]): # type: ignore
                    act_dict.update({actions: (self.scuttleAction, [x,y])})
                    actions += 1

        #Ace special action: boardwipe
        for x in range(4):
            #13 cards per rank, we are looking for rank 0 (Ace)
            act_dict.update({actions: (self.aceAction, [13 * x])})
            actions += 1

        for x in range(4):
            #13 cards per rank, we are looking for rank 4 (Five)
            act_dict.update({actions: (self.fiveAction, [(13 * x) + 4])})
            actions += 1
        return act_dict, actions

    def generateActionMask(self):
        inhand = np.where(self.current_zones["Hand"])
        onfield = np.where(self.off_zones["Field"])
        valid_actions = []

        for act_index, move in self.action_to_move.items():
            moveType = move[0]
            args = move[1]
            if  moveType == self.drawAction:
                valid_actions.append(act_index)
            elif moveType == self.scoreAction:
                card = args
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.scuttleAction:
                card = args[0]
                if card in inhand[0]:
                    target = args[1]

                    cRank = self.card_dict[card]["rank"]
                    cSuit = self.card_dict[card]["suit"]

                    tRank = self.card_dict[target]["rank"]
                    tSuit = self.card_dict[card]["suit"]

                    if onfield[0].size > 0 and target in onfield[0] and (cRank > tRank or (cRank == tRank and cSuit > tSuit)):
                        valid_actions.append(act_index)
            elif moveType == self.aceAction:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)
            elif moveType == self.fiveAction:
                card = args[0]
                if card in inhand[0]:
                    valid_actions.append(act_index)

        return valid_actions

    #Cards are generated as follows:
    #Generate all cards, in order (Ace = 0, King = 12), in a suit, then increase the suit
    #Ex. 0, 13, 26, and 39 are aces. Any card index is 13 * suit + rank
    def generateCards(self):
        cards = {}
        index = 0
        for suit in range(4):
            for rank in range(13):
                c ={"rank": rank, "suit":suit}
                cards.update({index:c})
                index += 1

        return cards

    def scoreState(self) -> int:
        field_scored = self.current_zones["Field"]
        index = 0
        score = 0
        for _ in range(4):
            for rank in range(13):
                if field_scored[index]:
                    if rank == 12:
                        score += 7
                    score += rank + 1
                index += 1
        return score

    def passControl(self):
        if self.current_zones is self.player_zones:
            self.current_zones = self.dealer_zones
            self.off_zones = self.player_zones
            return

        if self.current_zones is self.dealer_zones:
            self.current_zones = self.player_zones
            self.off_zones = self.dealer_zones
            return

