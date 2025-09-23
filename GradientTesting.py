import math
import random
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np # type: ignore
import gymnasium as gym


from Card import Card
from Cuttle import Cuttle
from Input import Manual, Randomized
from Moves import Draw, Move, ScorePlay, ScuttlePlay
from Person import Player
from Zone import Hand # type: ignore


#sets up for agent play
class Agent(Player):
    def __init__(self, hand, name, env):
        super().__init__(hand, name)
        self.env = env
        
    def turn(self, zones):
        super().turn(zones)
        select = random.randint(0, self.moves.__len__() - 1)
        select = self.moves[select]
        act = []
        if isinstance(select, Draw):
            act = [0,0,0]
        elif isinstance(select, ScorePlay):
            act = [0, self.env.getIndex(select.card), 0]
        elif isinstance(select, ScuttlePlay):
            act = [1, self.env.getIndex(select.card), self.env.getIndex(select.target)]
        else:
            act = [0,0,0]
            
        ob, reward = self.env.step(act, zones)
        
        print(ob, reward)
    
    def draw(self,deck):
        drawn = super().draw(deck)
        for x in self.hand.cards:
            self.env.agentHand[self.env.getIndex(drawn)]
        


class CuttleEnvironment(gym.Env):
    
    def __init__(self):
        
        self.game = Cuttle(Manual(Hand(0), "dealer"), Agent(None, "player", self))
        
        self.action_space = gym.spaces.MultiDiscrete([2, 52, 51])
        self.observation_space = gym.spaces.MultiBinary([5,52])
        
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
        
        self.agentHand = np.array(self.agentHand)
        self.agentField = np.array(self.agentField)
        self.oppField = np.array(self.oppField)
        self.scrapPile = np.array(self.scrapPile)
        
        
        
    def reset(self, seed, options):
        super().reset(seed = seed)
        self.game.gameStart
        
        observation = self.get_obs()
        
        return observation
    
    def render(self):
        pass
    
    def get_obs(self):
        #returns observation of current state
        return {"aHand": self.agentHand, "aField": self.agentField, "oField": self.oppField, "scrap":  self.scrapPile}
    
    def step(self, action, zones):
        move: Move = self.convertToMove(action, zones)
        reward = 0
        if self.game.player.cleanUp(zones):
            reward = 1
        
        move.execute()
        self.envExecute(action)
        
        observation = self.get_obs()
        
        return observation, reward
    
    def convertToMove(self, action, zones) -> Move:
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
    def envExecute(self, action):
        if (action[0] == 0):
            if (action[1] != 0):
                self.agentHand[action[1]] = 0
                self.agentField[action[1]] = 1
            else:
                for x in self.game.pHand.cards:
                    self.agentHand[self.getIndex(x)] = 1
        elif (action[0] == 1):
            self.agentHand[action[1]] = 0
            self.scrapPile[action[1]] = 1
            self.oppField[action[2]] = 0
            self.scrapPile[action[1]] = 1
        
    def envStart(self):
        self.game.gameStart()
        
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    
def train(hidden_sizes = [32] , lr = 1e-2, epochs = 50, batch_size = 5000, render = False):
    env = CuttleEnvironment()

    obs_dim = env.observation_space.shape

    n_acts = env.action_space.shape

    logits_net = mlp(sizes = [obs_dim] + hidden_sizes + [n_acts])
    
    def get_policy(obs):
        logits = logits_net(obs)
        return torch.distributions.Categorical(logits)
    
    def get_action(obs):
        return get_policy(obs).sample() 
    
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    

    
#print(gym.spaces.Discrete(1379).sample())
    
    
