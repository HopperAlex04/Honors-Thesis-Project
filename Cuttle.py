# The game itself, when this file is run it should present options for playing against the algorithims or yourself.

# TODO: Gamestates
# TODO: Turns
# TODO: Hands
# TODO: Card Functionality

import torch.nn as nn # type: ignore
from typing import cast
from Agent import DQNAgent, DQNOne
from Person import Player
from Zone import Deck, Hand, Zone

class Cuttle():
    
    def __init__(self, d, p):
        self.dealer = d
        self.player = p
        
        self.currPlayer = self.player
        self.offPlayer = self.dealer
        
        self.scrap = Zone(0, None, [])
        
        self.dHand = None
        self.pHand = None
        
        if self.dealer is not None and self.player is not None:
            self.player.hand = Hand(self.player)
            self.dealer.hand = Hand(self.dealer)
            self.dHand = self.dealer.hand
            self.pHand = self.player.hand
        
        
        self.deck = Deck()
        
        self.pfield = Zone(2, self.player, [])
        self.dfield = Zone(2, self.dealer, [])
        
        self.zones = [self.deck, self.scrap, self.dHand, self.pHand, self.pfield, self.dfield]
    
    def gameStart(self):
        self.player.score = 0
        self.dealer.score = 0
        for x in range(0,6):
            self.dealer.draw(self.deck)
        
        for x in range(0,5):
            self.player.draw(self.deck)
        
        over = False
        self.zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
        
        for x in self.zones:
                if x is not self.deck and x is not self.scrap:
                    print(x)
        
        while (not over):
            
            if not self.deck.cards: break
            
            self.currPlayer = self.player
            self.offPlayer = self.dealer
            self.zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
            #print(self.zones)
            self.player.turn(self.zones)
            over = self.player.cleanUp(self.zones)
            for x in self.zones:
                if x is not self.deck and x is not self.scrap:
                    print(x)
            
            
            if isinstance(self.dealer, DQNAgent): self.dealer.updateModel(self.zones)
            
            if over: continue
            if not self.deck.cards: break
            
            self.zones = [self.dHand, self.dfield, self.pHand, self.pfield, self.scrap, self.deck]
            #print(self.zones)
            self.currPlayer = self.dealer
            self.offPlayer = self.player
            if not self.deck.cards: break
            self.dealer.turn(self.zones)
            over = self.dealer.cleanUp(self.zones)
            for x in self.zones:
                if x is not self.deck and x is not self.scrap:
                    print(x)
                    
            if isinstance(self.player, DQNAgent): self.player.updateModel(self.zones)
            
        #print(self.pfield.cards[0])
        #print(self.dfield.cards[0])
        
        # this is for testing the computation of all actions
                    

            
        