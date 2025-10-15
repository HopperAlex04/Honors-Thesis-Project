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
            