from abc import ABC, abstractmethod
import random


class Player(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def getAction(self, validActions):
        pass
    
    def getReward(self, ob, score, terminated):
        if terminated and score >= 21: return 1


class Randomized(Player):
    def __init__(self, name):
        super().__init__(name)
        self.start = True
        self.prevOB = []
        self.currOB = []
        self.prevScoreDiff = 0
        self.currScoreDiff = 0

    def getAction(self, validActions:list) -> int:
        return validActions[random.randint(0, validActions.__len__() - 1)]
    
    def getReward(self, ob, scoreDiff, terminated):
        self.currOB = self.prevOB
        self.prevOB = ob
        
        delta = scoreDiff - self.prevScoreDiff
        
        self.prevScoreDiff = scoreDiff
        
        print(delta)
        return delta
        
        
        