#Treat the game as an arbitrary generator of states based on input actions, setting up a turn order as follows
from GameEnvironment import CuttleEnvironment


def trainingLoop(episodes):
    env = CuttleEnvironment()

    for x in range(episodes):
        env.reset()
        
        terminated = False

        while not terminated:
            #Get an action from the model
            action = 0
            ob, score, terminated = env.step(action)
            #This could probably be part of the step function
            if terminated:break
            env.passControl()
            #Get a reward based on the data from step and feed the training alg
            
            #Get an action from the model
            action = 0
            ob, score, terminated = env.step(action)
            if terminated:break
            #This could probably be part of the step function
            env.passControl()
            #Get a reward based on the data from step and feed the training alg
            
            