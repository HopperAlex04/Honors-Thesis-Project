from typing import cast
from Card import Card
from Moves import ScorePlay
import Moves
from Zone import Deck, Hand, Zone


class Player(): 
    
    def __init__(self, hand: Hand):
        self.hand = hand
        self.score = 0
        self.moves = []
        #self.zones = []
        #self.yard = yard
        
    def draw(self,deck):
        (cast(Hand, self.hand)).cards.append(cast(Deck, deck).cards.pop())
        
        #print((cast(Hand, self.hand)).cards)
        #print(cast(Deck, deck).cards)
        
    def computeMoves(self, card: Card, state: list):
        #add scoring moves to moves
        #due to the construction of turn ordering, the owners hand will always be 0, and field will always be 1
        self.moves.append(ScorePlay(card, cast(Hand, state[0]), cast(Zone, state[1])))
        pass
        
    def turn(self, zones: list):
        self.moves = []
        
        for x in self.hand.cards:
            self.computeMoves(x, zones)
        cast(ScorePlay, self.moves[0]).execute()
        pass
    
    def cleanUp(self, zones:list) -> bool:
        self.score = 0
        for x in cast(Zone, zones[1]).cards:
            self.score += cast(Card, x).number
        print(self.score)
        return self.score >= 21
    
    