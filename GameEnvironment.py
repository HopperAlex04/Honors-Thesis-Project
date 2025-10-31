import random
from typing import Optional
import numpy as np
import gymnasium as gym



class CuttleEnvironment(gym.Env):
    
    
    def __init__(self) -> None:
        
        self.dealerHand = np.zeros(52, dtype=bool)
        self.dealerField = np.zeros(52, dtype=bool)
        self.playerField = np.zeros(52, dtype=bool)
        self.playerHand = np.zeros(52, dtype=bool)
        self.deck = np.ones(52, dtype=bool)
        self.scrap = np.zeros(52, dtype=bool)
        
        self.playerZones = {"Hand": self.playerHand, "Field": self.playerField}
        self.dealerZones = {"Hand": self.dealerHand, "Field": self.dealerField}
        
        self.currentZones = self.playerZones
        self.offZones = self.dealerZones
        
        self.cardDict = self.generateCards()
        
        self.action_to_move, self.actions = self.generateActions()
        
        self.mask = []
        
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
            
    def step(self, action):
        act = self.action_to_move.get(action)
        if act is None: return None, 0, False, True 
        func = act[0] # type: ignore
        args = act[1] # type: ignore
        func(args)
        ob = self._get_obs()
        score = self.scoreState()
        terminated = score >= 21
        truncated = False
        
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
        scrap[card] = False
        for x in range(oppField.size): # type: ignore
            oppField[x] = False # type: ignore
            scrap[x] = True
    
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
            act_dict.update({actions: (self.aceAction, [x])})
            actions += 1
            
        return act_dict, actions
    
    def generateActionMask(self):
        handMask = np.where(self.currentZones["Hand"])
        fieldMask = np.where(self.offZones["Field"])
        fullMask = []
        for x in self.action_to_move:
            move = self.action_to_move[x]
            moveType = move[0]
            if  moveType == self.drawAction:
                fullMask.append(x)
            elif moveType == self.scoreAction:
                card = move[1]
                if card in handMask[0]:
                    fullMask.append(x)
            elif moveType == self.scuttleAction:
                card = move[1][0]
                if card in handMask[0]:
                    target = move[1][1]
                    
                    cRank = self.cardDict[card]["rank"]
                    cSuit = self.cardDict[card]["suit"]
                    
                    tRank = self.cardDict[target]["rank"]
                    tSuit = self.cardDict[card]["suit"]
                    
                    if fieldMask[0].size > 0 and target in fieldMask[0] and (cRank > tRank or (cRank == tRank and cSuit > tSuit)):
                        fullMask.append(x)
            elif moveType == self.aceAction:
                card = move[1][0]
                if card in handMask[0]:
                    fullMask.append(x)            
                        
            
        return fullMask
    
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
                if fieldScored[index]: score += rank
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
        
         