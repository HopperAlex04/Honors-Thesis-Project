from collections import deque, namedtuple
import os
import random
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


        