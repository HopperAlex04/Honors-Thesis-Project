import math
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np # type: ignore
import gymnasium as gym


from Card import Card
from Cuttle import Cuttle
from Input import Randomized
from Moves import Draw, Move, ScorePlay, ScuttlePlay
from Person import Player
from Zone import Hand # type: ignore


class CuttleEnvironment(gym.Env):
    
    def __init__(self):
        
        self.game = Cuttle(Randomized(Hand(0), "dealer"), Agent(None, "player", None, None, None))
        
        self.action_space = gym.spaces.MultiDiscrete([2, 52, 51])
        self.observation_space = gym.spaces.MultiBinary([4,52])
        
        #Setting up open boardstate
        self.agentHand = []
        self.agentField = []
        self.oppField = []
        self.scrapPile = []
        
        for x in range(52):
            self.agentHand.append(0)
            self.agentField.append(0)
            self.oppField.append(0)
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
    
    def step(self, action):
        move: Move = self.convertToMove(action)
        
        if self.game.player.cleanUp(self.game.zones):
            reward = 1
        
        move.execute()
        
        observation = self.get_obs()
        
        return observation, reward
    
    def convertToMove(self, action) -> Move:
        final: Move = Draw(self.game.pHand, self.game.deck)
        
        if (action[0] == 0):
            if (action[1] != 0):
                final = ScorePlay(self.getCard(action[1]), self.game.pHand, self.game.pfield)
            else:
                final = Draw(self.game.pHand, self.game.deck)
                print("drawing")
        elif (action[0] == 1):
            final = ScuttlePlay(self.getCard(action[1]), self.getCard(action[2] + 1), self.game.pHand, self.game.dfield, self.game.scrap)
        
        return final

#mode = 1 is hand, mode = 2 is target

    def getCard(self, index) -> Card:
        number = math.ceil(index/4)
        suit = index % 4
        match suit:
            case 1: suit = 1
            case 2: suit = 2
            case 3: suit = 3
            case 0: suit = 4
        
        selected = Card(number, suit)
        print(selected)
        for x in self.game.pHand.cards:
            if selected.number == x.number and selected.suit == x.suit:
                selected = x
                
        return selected
    
        
#sets up for agent play
class Agent(Player):
    def __init__(self, hand, name, aField, oField, scrap):
        super().__init__(hand, name)
        self.aField = aField
        self.oField = oField
        self.scrap = scrap
        
    def turn(self):
        pass
    
    
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
    

    

    

    
    
