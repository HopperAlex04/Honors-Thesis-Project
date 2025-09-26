import random
from Person import Player
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np # type: ignore


class Agent(Player):
    def __init__(self, hand, name, env):
        super().__init__(hand, name)
        self.env = env
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
        #print(f"Using {self.device} device")

        self.model = NeuralNetwork().to(self.device)
        #print(self.model)
        
    def turn(self, zones):
        super().turn(zones)
        
        self.env.envExecute(zones)
        valid = False
        
        #obs = self.env.get_obs
        state = np.concatenate((self.env.agentHandarr, self.env.agentFieldarr, self.env.oppFieldarr, self.env.scrapPilearr), axis = 0)
                
        stateT = torch.from_numpy(np.array([state])).float().to(device=self.device)
        #print(stateT)
        logits = self.model(stateT)
        
        mask = self.generateMask(zones)
        
        for x in range(1379):
            if x not in mask:
                logits[0, x] = float('-inf')
        
        
        probs = torch.softmax(logits, dim = 1)
        
        while not valid:
            pred_probab = nn.Softmax(dim=1)(probs)
            y_pred = pred_probab.argmax(1)
            mv = self.env.convertToMove(y_pred.item(), zones)
            print(y_pred.item())
            for x in self.moves:
                valid = x.__eq__(mv)
                if valid: break
                
        
        
            
        ob, reward = self.env.step(y_pred.item(), zones)
        
        #print(ob, reward)
    
    def draw(self,deck):
        drawn = super().draw(deck)
        for x in self.hand.cards:
            self.env.agentHandarr[self.env.getIndex(drawn)]
            
    def generateMask(self, zones):
        computedMoves = []               
        for x in range(1379):
            computedMoves.append(self.env.convertToMove(x, zones))
        
        indexes = []

        for x in self.moves:
            indexes.append(computedMoves.index(x))
            
        return indexes

            
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(52*4, 1517),
            nn.ReLU(),
            nn.Linear(1517, 1517),
            nn.ReLU(),
            nn.Linear(1517, 1379),
        )

    def forward(self, x):

        x = self.flatten(x)
        #print(x)
        logits = self.linear_relu_stack(x)
        return logits