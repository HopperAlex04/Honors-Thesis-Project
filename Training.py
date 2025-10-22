#Treat the game as an arbitrary generator of states based on input actions, setting up a turn order as follows
from abc import ABC, abstractmethod
import random
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork, get_action


class Randomized():
    def __init__(self, name):
        self.name = name

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
            #Get an action from the player and do it
            validActions = env.generateActionMask()
            action = p1.getAction(validActions)
            ob, score, terminated = env.step(action)
            
            #ob, score, terminated, and truncated (when I implement it), would be given to a model
            
            #Example: p1.updateModel(ob, score, terminated, truncated)
            
            #This could probably be part of the step function
            env.render()
            env.passControl()
            
def winReward01(episodes):
    env = CuttleEnvironment()
    actions = env.actions
    p1 = Randomized("p1")
    p2 = NeuralNetwork(52 * 5, actions)
    
    wins = 0
    
    with open("data.txt", "w") as f:
        f.write(f"Win rate per 10 episodes\n")
    for x in range(episodes):
        env.reset()
        
        terminated = False

        while not terminated:
            #Get an action from the player and do it
            validActions = env.generateActionMask()
            action = p1.getAction(validActions)
            ob, score, terminated = env.step(action)
            if terminated: break
            
            env.render()
            env.passControl()
            
            validActions = env.generateActionMask()
            action = get_action(p2, ob, validActions, actions)
            ob, score, terminated = env.step(action)
            if terminated:
                wins += 1
                break
            print(terminated)
            
            env.render()
            env.passControl()
            
        # Do learning stuff here
        
        if x != 0 and x % 10 == 0:
            with open("data.txt", "a") as f:
                f.write(f"{wins / x}\n")
            
