# The game itself, when this file is run it should present options for playing against the algorithims or yourself.

# TODO: Gamestates
# TODO: Turns
# TODO: Hands
# TODO: Card Functionality

from typing import cast
from Person import Player
from Zone import Deck, Hand, Zone

class Cuttle():
    
    def __init__(self, d: Player, p: Player):
        self.dealer = d
        self.player = p
        
        self.scrap = Zone(0, None, [])
        
        self.player.hand = Hand(self.player)
        self.dealer.hand = Hand(self.dealer)
        self.dHand = self.dealer.hand
        self.pHand = self.player.hand
        
        self.deck = Deck()
        
        self.pfield = Zone(2, self.player, [])
        self.dfield = Zone(2, self.dealer, [])
        
        self.zones = [self.deck, self.scrap, self.dHand, self.pHand, self.pfield, self.dfield]
    
    def gameStart(self):
        for x in range(0,6):
            self.dealer.draw(self.deck)
        
        for x in range(0,5):
            self.player.draw(self.deck)
        
        over = False
        
        while (not over):
            self.zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
            #print(self.zones)
            self.player.turn(self.zones)
            over = self.player.cleanUp(self.zones)
            if over: continue
            
            self.zones = [self.dHand, self.dfield, self.pHand, self.pfield, self.scrap, self.deck]
            #print(self.zones)
            self.dealer.turn(self.zones)
            over = self.dealer.cleanUp(self.zones)
            for x in self.zones:
                if x is not self.deck and x is not self.scrap:
                    print(x)
            
        #print(self.pfield.cards[0])
        #print(self.dfield.cards[0])
            
        