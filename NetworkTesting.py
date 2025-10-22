import torch
import numpy as np
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork, get_action
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
    mask = env.generateActionMask()
    y_pred = get_action(model, ob, mask, actions)
    print(y_pred.item()) # type: ignore
    print(mask)
    
# def buildModelTest():
#     m1 = build_model(10, [])
#     print(m1)
    