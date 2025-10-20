import torch
import numpy as np
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork
from torch import nn


def getActionTest():
    #Ensures model only outputs valid action after a mask is applied to the logits
    
    env = CuttleEnvironment()
    actions = env.actions
    print(actions)
    model = NeuralNetwork(52 * 5, env.actions)
    env.reset()
    env.render()
    
    ob = env._get_obs()
    state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
    stateT = torch.from_numpy(np.array([state])).float()
    
    mask = env.generateActionMask()
    logits = model(stateT)
    
    for x in range(actions):
            if x not in mask:
                logits[0, x] = float('-inf')
                
    probs = torch.softmax(logits, dim = 1)
    pred_probab = nn.Softmax(dim=1)(probs)
    y_pred = pred_probab.argmax(1)
    print(y_pred.item())
    print(mask)
    
    
    