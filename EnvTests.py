import numpy as np

from GameEnvironment import CuttleEnvironment

# from Move import DrawMove


def initTest():
    env = CuttleEnvironment()

    ob = env.get_obs()

    for x in ob:
        print(f"{x}: {ob.get(x)}")
        print()

    for x in env.action_to_move:
        act = env.action_to_move.get(x)
        args = act[1]  # type: ignore
        act = act[0]  # type: ignore
        print(act(args))
    print(env.actions)


def drawTest():
    env = CuttleEnvironment()

    act = env.action_to_move[0]
    func = act[0]
    args = act[1]

    func(args)

    ob = env.get_obs()
    #  print(ob)

    env.passControl()
    func()

    #  print(ob)
    for _x in np.where(env.deck)[0]:
        env.passControl()
        func()
    ob = env.get_obs()
    print(ob)


def scoreTest():
    env = CuttleEnvironment()

    draw = env.action_to_move.get(0)
    args = draw[1]  # type: ignore
    draw = draw[0]  # type: ignore

    for _ in env.deck:
        draw(args)

    for x in range(1, 53):
        score = env.action_to_move.get(x)
        args = score[1]  # type: ignore
        score = score[0]  # type: ignore
        score(args)
    ob = env.get_obs()
    print(ob)

    env = CuttleEnvironment()

    env.passControl()

    draw = env.action_to_move.get(0)
    args = draw[1]  # type: ignore
    draw = draw[0]  # type: ignore

    for _ in env.deck:
        draw(args)

    for x in range(1, 53):
        score = env.action_to_move.get(x)
        args = score[1]  # type: ignore
        score = score[0]  # type: ignore
        score(args)
    ob = env.get_obs()
    print(ob)


def resetTest():
    env = CuttleEnvironment()

    env.reset(None, None)

    if not len(np.where(env.dealer_hand)[0]) == 6:
        print(f"error {len(np.where(env.dealer_hand)[0])}")

    if not len(np.where(env.player_hand)[0]) == 5:
        print(f"error {len(np.where(env.dealer_hand)[0])}")


def generateCardsTest():
    env = CuttleEnvironment()
    print(env.card_dict)


def scuttleTest():
    env = CuttleEnvironment()
    env.player_hand[2] = True
    env.dealer_field[0] = True

    env.scuttleAction([2, 0])

    print(f"Card: {env.player_hand[2]}")
    print(f"Target: {env.dealer_field[0]}")
    print(f"Scrap 2:{env.scrap[2]}, 0: {env.scrap[0]}")


def maskTest():
    env = CuttleEnvironment()
    env.player_hand[0] = True
    env.player_hand[10] = True
    env.dealer_field[9] = True

    valid_actions = env.generateActionMask()
    print(valid_actions)
    for x in valid_actions:
        print(env.action_to_move[x])


def fiveTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][30] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.render()


def aceTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][0] = True
    env.current_zones["Field"][5] = True
    env.off_zones["Field"][6] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.render()


def twoTest():
    env = CuttleEnvironment()
    print(env.royal_indicies)
    env.current_zones["Hand"][1] = True
    env.off_zones["Field"][12] = True

    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.render()


def threeTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][2] = True
    env.scrap[12] = True

    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.render()


def sixTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][5] = True
    for x in env.royal_indicies:
        for index in x:
            env.off_zones["Field"][index] = True
    env.off_zones["Field"][7] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.render()


def sevenTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][6] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    print(env.effect_shown)
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[1])
    env.step(valid_actions[0])
    env.render()


def nineTest():
    env = CuttleEnvironment()
    env.current_zones["Hand"][8] = True
    env.off_zones["Field"][4] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[3])
    env.passControl()
    valid_actions = env.generateActionMask()
    print(valid_actions)


def fourTest():
    env = CuttleEnvironment()
    # Opp hand empty
    env.current_zones["Hand"][3] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.passControl()
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[0])
    env.render()

    env = CuttleEnvironment()
    # Opp hand 1
    env.current_zones["Hand"][3] = True
    env.off_zones["Hand"][5] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.passControl()
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[0])
    env.render()

    env = CuttleEnvironment()
    # Opp hand 2
    env.current_zones["Hand"][3] = True
    env.off_zones["Hand"][5] = True
    env.off_zones["Hand"][6] = True
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[2])
    env.passControl()
    valid_actions = env.generateActionMask()
    print(valid_actions)
    env.step(valid_actions[0])
    env.render()


fourTest()
