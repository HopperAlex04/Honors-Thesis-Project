from numpy import long
import torch
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork

from Players import Agent, HueristicHighCard, Randomized, Transition
import Players
import Training


def getActionTest():
    #Ensures model only outputs valid action after a mask is applied to the logits

    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, env.actions, None)

    BATCH_SIZE = 1
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    ag = Agent("player", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    env.reset()
    env.render()

    ob = env.get_obs()
    mask = env.generateActionMask()
    y_pred = ag.getAction(ob, mask, actions, 10000)



def trainingTest():
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, actions, None)

    BATCH_SIZE = 8192
    GAMMA = 0.7
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    p1 = Agent("Agent01", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    #p2 = Agent("dealer", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    p3 = HueristicHighCard("dealer")

    t = Training.WinRewardTraining(p1, p3)
    t.trainLoop(1500)



    t.validLoop(p1, p3, True, 1000)

def getStateTest():
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, actions, None)

    BATCH_SIZE = 4096
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    p1 = Agent("Agent01", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )

    env.reset()
    ob = env.get_obs()
    print(ob)
    state = p1.get_state(ob)
    print(state)
    print(len(state))
    print(state.dim())

def heur1Test():
    p1 = Players.HueristicHighCard("H1")
    env = CuttleEnvironment()

    env.reset()

    return p1.getAction(env.get_obs(), env.generateActionMask())

def optPrepTest():
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, actions, None)

    BATCH_SIZE = 4096
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    p1 = Agent("Agent01", model, 2, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )

    env.reset()
    ob = env.get_obs()

    state = p1.get_state(ob)
    action = 0
    env.step(0)
    ob = env.get_obs()
    next_state = p1.get_state(ob)

    p1.memory.push(state, torch.tensor([action]), next_state, torch.tensor([0]))
    p1.memory.push(state, torch.tensor([action]), next_state, torch.tensor([1]))
    p1.optimize()

def multiAgentTest():
    env = CuttleEnvironment()
    actions = env.actions
    model = NeuralNetwork(260, actions, None)
    torch.save(model, "./models/base.pt")
    BATCH_SIZE = 4096
    GAMMA = 0.5
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    p1 = Agent("Agent01", torch.load("./models/base.pt", weights_only=False), BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    p2 = Agent("Agent02", torch.load("./models/base.pt", weights_only=False), BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    p3 = Agent("Agent03", torch.load("./models/base.pt", weights_only=False), BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    p4 = Agent("Agent04", torch.load("./models/base.pt", weights_only=False), BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR )
    agents = [p1, p2,p3,p4]
    testP = HueristicHighCard("theOpp")
    t = Training.WinRewardTraining(agents[0], agents[0])
    for epochs in range(30):
        win_rates = []
        for a in agents:
            t = Training.WinRewardTraining(a, a)
            t.trainLoop(500)
            win_rates.append([0,0,0,0])

        for x in range(len(agents)):
            for y in range(len(agents)):
                if x != y:
                    p1WR, p2WR = t.validLoop(agents[x], agents[y])
                    win_rates[x][y] = p1WR
                    win_rates[y][x] = p2WR

        highWR = 0
        best = agents[0]
        for x in range(len(win_rates)):
            sum = 0
            for y in range(len(win_rates[0])):
                sum += win_rates[x][y]
            avg = sum / len(win_rates[0])
            if highWR < avg:
                highWR = avg
                best = agents[x]


        #t.validLoop(best, testP, 500)
        input("enter to continue")
        #t.validLoop(testP, best, 500)