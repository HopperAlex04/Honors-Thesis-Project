#Treat the game as an arbitrary generator of states based on input actions, setting up a turn order as follows
from abc import ABC, abstractmethod
import random
from GameEnvironment import CuttleEnvironment


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

    def getAction(self, validActions:list) -> int:
        return validActions[random.randint(0, validActions.__len__() - 1)]


#Loop using randomized action generation for testing
def randomLoop(episodes):
    env = CuttleEnvironment()

    p1 = Randomized("p1")
    for x in range(episodes):
        env.reset()
        
        terminated = False

        while not terminated:
            #Get an action from the model
            validActions = env.generateActionMask()
            action = p1.getAction(validActions)
            ob, score, terminated = env.step(action)
            #This could probably be part of the step function
            env.render()
            env.passControl()
            
