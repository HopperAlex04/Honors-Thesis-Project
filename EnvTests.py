import numpy as np
from GameEnvironment import CuttleEnvironment
#from Move import DrawMove


def initTest():
    env = CuttleEnvironment()

    ob = env._get_obs()

    for x in ob:
        print(f"{x}: {ob.get(x)}")
        print()

    for x in env.action_to_move:
        act = env.action_to_move.get(x)
        args = act[1] # type: ignore
        act = act[0] # type: ignore
        print(act(args))
    print(env.actions)

def drawTest():
    env = CuttleEnvironment()

    act = env.action_to_move[0]
    func = act[0]
    args = act[1]

    func(args)

    ob = env._get_obs()
    #print(ob)

    env.passControl()
    func()

    #print(ob)
    for x in np.where(env.deck)[0]:
        env.passControl()
        func()
    ob = env._get_obs()
    print(ob)

def scoreTest():
    env = CuttleEnvironment()

    draw = env.action_to_move.get(0)
    args = draw[1] # type: ignore
    draw = draw[0] # type: ignore

    for x in env.deck:
        draw(args)

    for x in range(1, 53):
        score = env.action_to_move.get(x)
        args = score[1] # type: ignore
        score = score[0] # type: ignore
        score(args)
    ob = env._get_obs()
    print(ob)

    env = CuttleEnvironment()

    env.passControl()

    draw = env.action_to_move.get(0)
    args = draw[1] # type: ignore
    draw = draw[0] # type: ignore

    for x in env.deck:
        draw(args)

    for x in range(1, 53):
        score = env.action_to_move.get(x)
        args = score[1] # type: ignore
        score = score[0] # type: ignore
        score(args)
    ob = env._get_obs()
    print(ob)

def resetTest():
    env = CuttleEnvironment()

    env.reset()

    if not len(np.where(env.dealerHand)[0]) == 6:
        print(f"error {len(np.where(env.dealerHand)[0])}")

    if not len(np.where(env.playerHand)[0]) == 5:
        print(f"error {len(np.where(env.dealerHand)[0])}")

def generateCardsTest():
    env = CuttleEnvironment()
    print(env.cardDict)

def scuttleTest():
    env = CuttleEnvironment()
    env.playerHand[2] = True
    env.dealerField[0] = True

    env.scuttleAction([2, 0])

    print(f"Card: {env.playerHand[2]}")
    print(f"Target: { env.dealerField[0]}")
    print(f"Scrap 2:{env.scrap[2]}, 0: {env.scrap[0]}")

def maskTest():
    env = CuttleEnvironment()
    env.playerHand[0] = True
    env.playerHand[10] = True
    env.dealerField[9] = True

    mask = env.generateActionMask()
    print(mask)
    for x in mask:
        print(env.action_to_move[x])

def fiveTest():
    env = CuttleEnvironment()
    env.currentZones["Hand"][30] = True
    mask = env.generateActionMask()
    print(mask)
    env.step(mask[2])
    env.render()