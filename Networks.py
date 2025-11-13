from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, obspace, embedding_size, actions: int, seq: nn.Sequential|None):
        super().__init__()
        #self.flatten = nn.Flatten()
        if seq:
            self.linear_relu_stack = seq
        else:
            input_length = 0
            for item in obspace.values():
                if isinstance(item, dict):
                    for x in item.values():
                        input_length += len(x)
                if isinstance(item, np.ndarray):
                    input_length += len(item)
                    print(input_length)
                elif isinstance(item, list):
                    for _ in item:
                        input_length += embedding_size
                        print(input_length)
            self.embedding = nn.Embedding(53, embedding_size) # 0 indicates nothing is there, 1-52 are cards
            self.linear_relu_stack = nn.Sequential(nn.Linear(input_length, actions), nn.Tanh())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits