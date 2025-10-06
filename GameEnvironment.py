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
        
        self.action_to_move, self.actions = self.generateActions()
        
        self.observation_space = gym.spaces.MultiBinary([6,52])
        self.action_space = gym.spaces.Discrete(self.actions)
        
        
        
    def _get_obs(self):
        return {"Dealer Hand": self.dealerHand, "Dealer Field": self.dealerField, "Player Field": self.playerField, 
                "Player Hand": self.playerHand, "Deck": self.deck, "Scrap": self.scrap}
        
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
        func = act[0] # type: ignore
        args = act[1] # type: ignore
    
    def drawAction(self, *args):
        hand = self.currentZones.get("Hand")
        
        possibleDraws = np.where(self.deck)[0]
        if possibleDraws.any():
            index = possibleDraws[random.randint(0, len(possibleDraws) - 1)]
            hand[index] = True # type: ignore
            self.deck[index] = False
        else: return 1
        
        return "Draw"
    
    def scoreAction(self, *args):
        hand = self.currentZones.get("Hand")
        field = self.currentZones.get("Field")
        card = args[0]
        
        hand[card] = False # type: ignore
        field[card] = True # type: ignore
    
    def scuttleAction(self, *args):
        hand = self.currentZones.get("Hand")
        oppField = self.offZones.get("Field")
        scrap = self.scrap
        
        
        card = args[0]
        target = args[1]
        
        hand[card] = False # type: ignore
        oppField[target] = False # type: ignore
        scrap[card] = False
        scrap[target] = False
        
            
    
    def generateActions(self):
        #Initializes storage mediums
        act_dict = {}
        actions = 0
        
        #Adds draw action
        act_dict.update({0: (self.drawAction, "")})
        actions += 1
        
        while actions < 53:
            act_dict.update({actions: (self.scoreAction, actions - 1)})
            actions += 1
        
        
            
        return act_dict, actions
    
    
    def passControl(self):
        if self.currentZones is self.playerZones:
            self.currentZones = self.dealerZones
            self.offZones = self.playerZones
            return
        
        if self.currentZones is self.dealerZones:
            self.currentZones = self.playerZones
            self.offZones = self.dealerZones
            return
        
        