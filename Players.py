from abc import ABC, abstractmethod
import random


class Player(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def getAction(self, state, mask):
        return 0
    
class Randomized(Player):
    def __init__(self, name, seed = 0):
        super().__init__(name)
        if seed != 0: random.seed(seed)
        
    def getAction(self, state, mask):
        return mask[random.randint(0, mask.__len__() - 1)]
    
class Agent(Player):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
    
    def getAction(self, state, mask):
        #the hard part
        pass
        