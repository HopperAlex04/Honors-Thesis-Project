from collections import deque, namedtuple
import os
import random
import numpy as np
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, obspace: int, actions: int, seq: nn.Sequential|None):
        super().__init__()
        #self.flatten = nn.Flatten()
        if seq:
            self.linear_relu_stack = seq
        else:
            self.linear_relu_stack = nn.Sequential(nn.Linear(obspace, actions), nn.Tanh())
            
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

# net = NeuralNetwork(52 * 5, 1379, None)
# print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0])

# input = torch.randn(52 * 5)
# out = net(input)
# print(out)

# net.zero_grad()
# out.backward(torch.randn(1379))