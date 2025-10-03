#zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
from typing import cast
import matplotlib
import matplotlib.pyplot as plt

import torch
from Agent import DQNAgent, DQNOne, ReplayMemory
from Card import Card
from Environment import CuttleEnvironment
from Input import Manual, Randomized
from Moves import Draw, ScorePlay, ScuttlePlay
from Zone import Hand, Zone

def actionToMoveTest():
    env = CuttleEnvironment(Manual(Hand(0), "dealer"), Manual(Hand(0), "dealer"))
    cards = []

    zones = [Hand(None), Zone(0, None,[]), Zone(0, None,[]), Zone(0, None,[]), Zone(0, None,[]), Zone(0, None,[])]
    for x in range(13):
        for y in range(4):
            inCard = Card(x+1, y+1)
            cards.append(inCard)
            for z in zones:
                z.cards.append(inCard)
            
            
    moves = []

    moves.append(Draw(zones[0], zones[5]))

    for x in cards:
        moves.append(ScorePlay(x, zones[0], zones[1]))

    sdrac = []
    cards2 = []


    for x in cards:
        cards2.append(x)
        sdrac.append(cards.pop())


    for x in cards2:
        
        for y in sdrac:
            if y.number <= x.number:
                if y.number == x.number:
                    if y.suit < x.suit: moves.append(ScuttlePlay(x, y, zones[0], zones[3], zones[4]))
                else:
                    moves.append(ScuttlePlay(x, y, zones[0], zones[3], zones[4]))
                    
    computedMoves = []               
    for x in range(1379):
        computedMoves.append(env.convertToMove(x, zones))
    #print(cast(ScuttlePlay, env.convertToMove(53, zones)).card)
    #print(env.getCard(0, zones[0]).__str__())

    #print(zones[0])

    i = 0
    #print(moves[0].__eq__(computedMoves[0]))
    for x in moves:
        if not (x.__eq__(computedMoves[i])):
            print("error")
            print(i)
            print(x, computedMoves[i])
            print(x.card, computedMoves[i].card)
            if isinstance(x, ScuttlePlay):
                print(x.target, computedMoves[i].target)
            print(x.card == computedMoves[i].card)
            print(x.hand == computedMoves[i].hand)
            print(x.field == computedMoves[i].field)
            break
        i += 1
    print("done")

datap = []
datad = []

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('ScoreGap')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def AgentTest(episodes):
    env = CuttleEnvironment(None, None)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
    memory = ReplayMemory(10000)
    network = DQNOne(52*4, 1379).to(device)
    
    env.player = DQNAgent(Hand(0), "player", env, network, memory, device, datap) # type: ignore
    #env.dealer = DQNAgent(Hand(0), "dealer", env, network, memory, device, datad) # type: ignore
    env.dealer = Randomized(Hand(0), "dealer") # type: ignore
    
    env.game.player = env.player # type: ignore
    env.game.dealer = env.dealer # type: ignore
    
    env.game.dHand = env.dealer.hand # type: ignore
    env.game.pHand = env.player.hand # type: ignore
    
    
    
    
    #print(env.observation_space.shape[0])
    wins = 0
    for x in range(episodes):
        env.envStart()
        datad.append(env.game.dealer.score - env.game.player.score) # type: ignore
        datap.append(env.game.player.score - env.game.dealer.score) # type: ignore
        plot_durations(datap)
        env.reset()
        
    print('Complete')
    plot_durations(datap, show_result=True)
    #plot_durations(datad, show_result=True)
    plt.ioff()
    plt.show()
    print(datap)
    #print(datad)
    
    return network.state_dict()
        
#actionToMoveTest()
model1 = "./models/model1.pth"
torch.save(AgentTest(1000), model1)