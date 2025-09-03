# The game itself, when this file is run it should present options for playing against the algorithims or yourself.

# TODO: Gamestates
# TODO: Turns
# TODO: Hands
# TODO: Card Functionality

from typing import cast
from Person import Player
from Zone import Deck, Hand, Zone

class Cuttle():
    
    player: Player
    dealer: Player
    deck: Deck
    yardD: Zone
    yardP: Zone
    dHand: Hand
    pHand: Hand
    pfield: Zone
    dfield: Zone
    
    zones: list
    
    def __init__(self, d: Player, p: Player):
        self.dealer = d
        self.player = p
        
        self.yardD = Zone(2, self.dealer, [])
        self.yardP = Zone(1, self.player, [])
        
        self.player.hand = Hand(self.player)
        self.dealer.hand = Hand(self.dealer)
        self.dHand = self.dealer.hand
        self.pHand = self.player.hand
        
        self.deck = Deck()
        
        self.pfield = Zone(2, self.player, [])
        self.dfield = Zone(2, self.dealer, [])
        
        self.zones = [self.deck, self.yardD, self.yardP, self.dHand, self.pHand, self.pfield, self.dfield]
    
    def gameStart(self):
        for x in range(0,6):
            self.dealer.draw(self.deck)
        
        for x in range(0,5):
            self.player.draw(self.deck)
        
        over = False
        
        while (not over):
            self.player.turn(self.zones)
            over = self.player.cleanUp()
            self.dealer.turn(self.zones)
            over = self.dealer.cleanUp()
            
        