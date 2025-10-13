from GameEnvironment import CuttleEnvironment
from Players import Randomized


def randomGetActionTest():
    env = CuttleEnvironment()
    p1 = Randomized("player")
    
    env.playerHand[0] = True
    env.playerHand[40] = True
    env.dealerField[13] = True
    
    mask = env.generateActionMask()
    print(mask)
    
    for x in range(100):
        num = p1.getAction(mask)
        if num not in mask: 
            print("error")
            break
    