import torch
import numpy as np
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork
from torch import nn

from Players import Agent


def getActionTest():
    #Ensures model only outputs valid action after a mask is applied to the logits
    
    env = CuttleEnvironment()
    actions = env.actions
    print(actions)
    model = NeuralNetwork(52 * 5, env.actions, None)
    
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    ag = Agent("player", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    env.reset()
    env.render()
    
    ob = env._get_obs()
    mask = env.generateActionMask()
    y_pred = ag.getAction(ob, mask, actions, 10000)
    print(y_pred.item()) # type: ignore
    print(mask)
    
getActionTest()
    