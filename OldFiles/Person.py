import random
from typing import cast
from Card import Card
from Moves import AceAction, Draw, Move, ScorePlay, ScuttlePlay
import Moves
from Zone import Deck, Hand, Zone


class Player(): 
    
    def __init__(self, hand: Hand, name: str):
        self.hand = hand
        self.name = name
        self.score = 0
        self.moves = []
        #self.zones = []
        #self.yard = yard
        
    def draw(self,deck):
        drawn = cast(Deck, deck).cards.pop()
        (cast(Hand, self.hand)).cards.append(drawn)
        return drawn
        
        #print((cast(Hand, self.hand)).cards)
        #print(cast(Deck, deck).cards)
        
    def computeMoves(self, card: Card, zones: list):
        #self.zones = [self.pHand, self.pfield, self.dHand, self.dfield, self.scrap, self.deck]
        self.moves.append(ScorePlay(card, cast(Hand, zones[0]), cast(Zone, zones[1])))
        
        for x in zones[3].cards:
            if x.number < card.number or (x.number == card.number and x.suit < card.suit):
                scuttle = ScuttlePlay(card, x, self.hand, zones[3], zones[4])
                self.moves.append(scuttle)
                
        if card.number == 1:
            self.moves.append(AceAction(card, zones[0], zones[1], zones[3], zones[4]))
        
    def turn(self, zones: list):
        self.moves = []
        
        for x in self.hand.cards:
            self.computeMoves(x, zones)
        #print(self.moves.__len__())
        if zones[5].cards: self.moves.append(Draw(self.hand, zones[5]))
        
    
    def cleanUp(self, zones:list) -> bool:
        self.score = 0
        for x in cast(Zone, zones[1]).cards:
            self.score += cast(Card, x).number
        print(f"{self.name}: {self.score}")
        return self.score >= 21
    
    def getInput(self) -> str:
        return ""
    
    