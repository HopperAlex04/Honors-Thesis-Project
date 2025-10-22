import os
import numpy as np
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, obspace, actions: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obspace, actions//2),
            nn.ReLU(),
            nn.Linear(actions//2, actions),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def get_state(ob):
    state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
    stateT = torch.from_numpy(np.array([state])).float()
    return stateT

def get_win_reward(score):
    reward = 0
    if score >= 21: reward = 1
    
    return reward

def get_action(model, ob, mask, actions):
    stateT = get_state(ob)
    logits = model(stateT)
    
    for x in range(actions):
            if x not in mask:
                logits[0, x] = float('-inf')
                
    probs = torch.softmax(logits, dim = 1)
    pred_probab = nn.Softmax(dim=1)(probs)
    y_pred = pred_probab.argmax(1)
    return y_pred.item()
        