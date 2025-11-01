import random
from typing import Optional
import numpy as np
import gymnasium as gym



class CuttleEnvironment(gym.Env):
    
    #Initializes the environment and defines the observation and action spaces
    def __init__(self) -> None:
        
        #Generates the zones
        #A zone is a bool np array
        #Where a card is can be determined as follows: 13*suit + rank, where suit is 0-3 and rank is 0-12
        self.dealerHand = np.zeros(52, dtype=bool)
        self.dealerField = np.zeros(52, dtype=bool)
        self.playerField = np.zeros(52, dtype=bool)
        self.playerHand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)
        
        #Defines who owns what zones, allows for easy access to fields
        self.playerZones = {"Hand": self.playerHand, "Field": self.playerField}
        self.dealerZones = {"Hand": self.dealerHand, "Field": self.dealerField}
        
        #Swapped by passControl(), always start with the player
        self.currentZones = self.playerZones
        self.offZones = self.dealerZones
        
        #Generates the cards for easy access to rank and suit based on index (demonstrated above)
        self.cardDict = self.generateCards()
        
        #Generates the actions, as well as determining how many actions are in the environment.
        #Actions from the action_to_move dict are of the form (moveType, [args]), where moveType is one of the functions below.
        self.action_to_move, self.actions = self.generateActions()
        
        #Gym helps us out so we make gym spaces
        self.observation_space = gym.spaces.MultiBinary([6,52])
        self.action_space = gym.spaces.Discrete(self.actions)
         
    def _get_obs(self):
        #Slight abstraction here, the current zones are the current players field and hand, while off zones are the opposite player's hand and field
        #This allows passControl to affect what will be visible to who when turns or priority changes.
        return {"Current Zones": self.currentZones, "Off-Player Zones": self.offZones, "Deck": self.deck, "Scrap": self.scrap}
        
    def _get_info(self):
        #return {"Score Difference": np.linalg.norm(player.score - dealer.score)} is the basic idea, for later
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        
        #Reset to open state to make new game
        self.dealerHand = np.zeros(52, dtype=bool)
        self.dealerField = np.zeros(52, dtype=bool)
        self.playerField = np.zeros(52, dtype=bool)
        self.playerHand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)
        
        #Makes sure all the zones are in the right places
        self.playerZones = {"Hand": self.playerHand, "Field": self.playerField}
        self.dealerZones = {"Hand": self.dealerHand, "Field": self.dealerField}
        
        self.currentZones = self.playerZones
        self.offZones = self.dealerZones
        
        #Draw opening hands
        draw = self.action_to_move.get(0)
        args = draw[1] # type: ignore
        draw = draw[0] # type: ignore
        self.passControl()
        for x in range(6):
            draw(args)
            
        self.passControl()
        for x in range(5):
            draw(args)
    
    #Converts an action into a move by grabbing the calling the function with args from the move dict
    def step(self, action:int):
        act = self.action_to_move.get(action)
        
        #This is to prevent a crash in the event of exhausting all possible actions, for games this ends the game
        if act is None: return None, 0, False, True 
        func = act[0] # type: ignore
        args = act[1] # type: ignore
        func(args)
        ob = self._get_obs()
        score = self.scoreState()
        terminated = score >= 21
        truncated = False
        
        #ob is of the form [dict, dict] and should be broken up when reading a state
        return ob, score, terminated, truncated
    
    def render(self):
        currHand = self.currentZones["Hand"]
        currField = self.currentZones["Field"]
        index = 0
        zoneString = ""
        zoneString += "Current Hand"
        for suit in range(4):
            for rank in range(13):
                if currHand[index]: zoneString += f" |{rank} {suit}| "
                index += 1
        print(zoneString)
        
        index = 0
        zoneString = ""
        zoneString += "Current Field"
        for suit in range(4):
            for rank in range(13):
                if currField[index]: zoneString += f" |{rank} {suit}| "
                index += 1
        print(zoneString)
             
                
        offHand = self.offZones["Hand"]
        offField = self.offZones["Field"]
        index = 0
        zoneString = ""
        zoneString += "Off Field"
        for suit in range(4):
            for rank in range(13):
                if offField[index]: zoneString += f" |{rank} {suit}| "
                index += 1
        print(zoneString)
        
        index = 0
        zoneString = ""
        zoneString += "Off Hand"
        for suit in range(4):
            for rank in range(13):
                if offHand[index]: zoneString += f" |{rank} {suit}| "
                index += 1
        print(zoneString)
        
        index = 0
        zoneString = f"Scrap: "
        for suit in range(4):
            for rank in range(13):
                if self.scrap[index]: zoneString += f" |{rank} {suit}| "
                index += 1
        print(zoneString)
        print(f"Curr Player Score: {self.scoreState()}")
        
    
    def drawAction(self, *args):
        hand = self.currentZones.get("Hand")
        
        possibleDraws = np.where(self.deck)[0]
        if possibleDraws.any():
            index = possibleDraws[random.randint(0, len(possibleDraws) - 1)]
            hand[index] = True # type: ignore
            self.deck[index] = False
        else: return 1
        
        return "Draw"
    
    def scoreAction(self, card):
        hand = self.currentZones.get("Hand")
        field = self.currentZones.get("Field")
        
        
        hand[card] = False # type: ignore
        field[card] = True # type: ignore
        
        return f"Scored f{card}"
    
    def scuttleAction(self, cardAndTarget):
        hand = self.currentZones.get("Hand")
        oppField = self.offZones.get("Field")
        scrap = self.scrap
        
        
        card = cardAndTarget[0]
        target = cardAndTarget[1]
        
        hand[card] = False # type: ignore
        oppField[target] = False # type: ignore
        scrap[card] = True
        scrap[target] = True
        
        return f"Scuttled {target} with {card}"

    def aceAction(self, card):
        hand = self.currentZones.get("Hand")
        oppField = self.offZones.get("Field")
        scrap = self.scrap
        
        hand[card] = False # type: ignore
        scrap[card] = True
        for x in range(oppField.size): # type: ignore
            oppField[x] = False # type: ignore
            scrap[x] = True
    
    def fiveAction(self, card):
        hand = self.currentZones.get("Hand")
        scrap = self.scrap
        
        hand[card] = False # type: ignore
        scrap[card] = True
        
        self.drawAction()
        self.drawAction()
        
    def nineAction(self, cardtargetselfHit):
        currHand = self.currentZones.get("Hand")
        currField = self.currentZones.get("Field")
        offField = self.offZones.get("Field")
        offHand = self.offZones.get("Hand")
        scrap = self.scrap
        
        card = cardtargetselfHit[0]
        target = cardtargetselfHit[1]
        selfHit = cardtargetselfHit[2]
        
        currHand[card] = False # type: ignore
        scrap[card] = True
        
        if selfHit:
            currField[target] = False # type: ignore
            currHand[target] = True # type: ignore
        else:
            offField[target] = False # type: ignore
            offHand[target] = True # type: ignore
    
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
            cardUsed = self.cardDict[x] # type: ignore
            for y in range(52):
                target = self.cardDict[y] # type: ignore
                if target["rank"] < cardUsed["rank"] or (target["rank"] == cardUsed["rank"] and target["suit"] < cardUsed["suit"]): # type: ignore
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
            
        for x in range(4):
            #13 cards per rank, we are looking for rank 8 (Nine), one for each card
            for y in range(51):
                act_dict.update({actions: (self.nineAction, [(13 * x) + 8, y, True])})
                actions += 1
                act_dict.update({actions: (self.nineAction, [(13 * x) + 8, y, False])})
                actions += 1   
        return act_dict, actions
    
    
    def generateActionMask(self):
        inHand = np.where(self.currentZones["Hand"])
        selfField = np.where(self.currentZones["Field"])
        onField = np.where(self.offZones["Field"])
        oppHand = np.where(self.offZones["Hand"])
        validActions = []
        for x in self.action_to_move:
            move = self.action_to_move[x]
            moveType = move[0]
            if  moveType == self.drawAction:
                validActions.append(x)
            elif moveType == self.scoreAction:
                card = move[1]
                if card in inHand[0]:
                    validActions.append(x)
            elif moveType == self.scuttleAction:
                card = move[1][0]
                if card in inHand[0]:
                    target = move[1][1]
                    
                    cRank = self.cardDict[card]["rank"]
                    cSuit = self.cardDict[card]["suit"]
                    
                    tRank = self.cardDict[target]["rank"]
                    tSuit = self.cardDict[card]["suit"]
                    
                    if onField[0].size > 0 and target in onField[0] and (cRank > tRank or (cRank == tRank and cSuit > tSuit)):
                        validActions.append(x)
            elif moveType == self.aceAction:
                card = move[1][0]
                if card in inHand[0]:
                    validActions.append(x)
            elif moveType == self.fiveAction:
                card = move[1][0]
                if card in inHand[0]:
                    validActions.append(x)
            elif moveType == self.nineAction:
                card = move[1][0]
                if card in inHand[0]:
                    target = move[1][1]
                    selfHit = move[1][2]
                    
                    if selfHit and selfField[0].size > 0 and target in selfField[0]:validActions.append(x)
                    elif not selfHit and onField[0].size > 0 and target in onField[0]: validActions.append(x)    
                                   
                        
            
        return validActions
    
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
        fieldScored = self.currentZones["Field"]
        index = 0
        score = 0
        for suit in range(4):
            for rank in range(13):
                if fieldScored[index]: 
                    if rank == 12:
                        score += 7
                    score += rank + 1
                index += 1
        return score
    
    def passControl(self):
        if self.currentZones is self.playerZones:
            self.currentZones = self.dealerZones
            self.offZones = self.playerZones
            return
        
        if self.currentZones is self.dealerZones:
            self.currentZones = self.playerZones
            self.offZones = self.dealerZones
            return
        
         