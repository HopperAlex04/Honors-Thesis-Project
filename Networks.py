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