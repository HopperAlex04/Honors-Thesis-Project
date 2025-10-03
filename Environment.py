import math
import random
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np # type: ignore
import gymnasium as gym


#from Agent import Agent
from Card import Card
from Cuttle import Cuttle
#from Input import Manual, Randomized
from Moves import Draw, Move, ScorePlay, ScuttlePlay
#from Person import Player
from Zone import Hand # type: ignore


#sets up for agent play
class CuttleEnvironment(gym.Env):
    
    def __init__(self, dealer, player):
        
        
        self.dealer = dealer
        self.player = player
        
        self.game = Cuttle(self.dealer, self.player)
        
        self.action_space = gym.spaces.Discrete(1379)
        self.observation_space = gym.spaces.MultiBinary([4,52])
        
        #Setting up open boardstate
        self.agentHand = []
        self.agentField = []
        self.oppField = []
        self.oppHand = []
        self.scrapPile = []
        
        for x in range(52):
            self.agentHand.append(0)
            self.agentField.append(0)
            self.oppField.append(0)
            self.oppHand.append(0)
            self.scrapPile.append(0)
        
        self.agentHandarr = np.array(self.agentHand)
        self.agentFieldarr = np.array(self.agentField)
        self.oppFieldarr = np.array(self.oppField)
        self.scrapPilearr = np.array(self.scrapPile)
        
        
        
    def reset(self):
        super().reset(seed = 0)
        self.game = Cuttle(self.dealer, self.player)
        
        observation = self.get_obs()
        
        return observation
    
    def render(self):
        pass
    
    def get_obs(self):
        #returns observation of current state
        return {"aHand": self.agentHandarr, "aField": self.agentFieldarr, "oField": self.oppFieldarr, "scrap":  self.scrapPilearr}
    
    def step(self, action, zones):
        move: Move = self.convertToMove(action, zones)
        reward = 0
        reward = self.game.currPlayer.score - self.game.offPlayer.score
        
        #print(move)
        move.execute()
        self.envLoad(zones)
        
        observation = self.get_obs()
        
        return observation, reward
    
    def convertToMove(self, action, zones) -> Move:
        #print(action)
        final: Move = Draw(zones[0], zones[5])
        
        if action == 0:
            final = Draw(zones[0], zones[5])
            
        elif action in range(1, 53):
            scoreCard = self.getCard(action - 1, zones[0])
            final = ScorePlay(scoreCard, zones[0], zones[1])
            
        elif action in range(53, 1379):
            count = 0
            indices = []

            for x in range(52):
                count += 1
                
                
            for x in range(52):
                
                for y in range(x + 1):
                    count += 1
                indices.append(count)
            
            index = 1 
            #print(indices)
            for x in indices:
                #print(x)
                if action <= x:
                    break
                index += 1
                    
            if index > -1:
                pCard = self.getCard(index, zones[0])
                #print(index, indices[index - 1], action, self.getCard(indices[index - 1] - action, zones[3]))
                oCard = self.getCard(indices[index - 1] - action, zones[3])
                final = ScuttlePlay(pCard, oCard, zones[0], zones[3], zones[4])        
        
        return final

#mode = 1 is hand, mode = 2 is target

    def getCard(self, index, zone) -> Card:
        number = math.floor(index/4)
        suit = index - number * 4
        
        selected = Card(number + 1, suit + 1)
        #print(selected)
        
        #print(zone)
        for x in zone.cards:
            if x.number == selected.number and x.suit == selected.suit:
                selected = x
        
        #print(selected)
        return selected
    
    def getIndex(self, card:Card):
        return (4 * (card.number- 1)) + card.suit-1
    
    #Necessary since game and env are seperate
    def envLoad(self, zones):
        
        self.agentHandarr.fill(0)
        self.agentFieldarr.fill(0)
        self.oppFieldarr.fill(0)
        self.scrapPilearr.fill(0)
        
        for x in zones[0].cards:
            self.agentHandarr[self.getIndex(x)] = 1
        for x in zones[1].cards:
            self.agentFieldarr[self.getIndex(x)] = 1
            
        for x in zones[3].cards:
            self.oppFieldarr[self.getIndex(x)] = 1
            
        for x in zones[4].cards:
            self.scrapPilearr[self.getIndex(x)] = 1
        
    def envStart(self):
        self.game.gameStart()
        
    
    
