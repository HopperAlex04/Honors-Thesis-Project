import torch
import numpy as np
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork
from torch import nn

from Players import Agent, Randomized
import Training


def getActionTest():
    #Ensures model only outputs valid action after a mask is applied to the logits
    
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, env.actions, None)
    
    BATCH_SIZE = 1
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



def trainingTest():
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, actions, None)
    
    BATCH_SIZE = 4096
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    p1 = Agent("Agent01", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    #p2 = Agent("dealer", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    
    t = Training.WinRewardTraining(p1, p1)
    t.trainLoop(1000)
    
    p3 = Randomized("dealer")
    
    t.validLoop(p3, 1000)
    
    
    
getActionTest()
trainingTest()
    