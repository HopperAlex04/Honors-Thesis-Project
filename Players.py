from abc import ABC, abstractmethod
import random


class Player(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def getAction(self):
        pass


class Randomized(Player):
    def __init__(self, name):
        super().__init__(name)

    def getAction(self, validActions:list) -> int:
        return validActions[random.randint(0, validActions.__len__() - 1)]
        