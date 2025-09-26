#zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
from typing import cast

import torch
from Card import Card
from Environment import CuttleEnvironment
from Moves import Draw, ScorePlay, ScuttlePlay
from Zone import Hand, Zone

def actionToMoveTest():
    env = CuttleEnvironment()
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

def AgentTest():
    env = CuttleEnvironment()
    #print(env.observation_space.shape[0])
    for x in range(10):
        env.envStart()
        env.reset()
        
#actionToMoveTest()
AgentTest()