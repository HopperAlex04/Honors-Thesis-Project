from GameEnvironment import CuttleEnvironment
from Players import Randomized


def randomGetActionTest():
    p1 = Randomized("player")
    p2 = Randomized("dealer")
    
    env = CuttleEnvironment(p1, p2)
    
    
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
    