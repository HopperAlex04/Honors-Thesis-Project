from typing import cast
from Zone import Deck, Hand


class Player():
    
    score = 0
    def __init__(self, hand):
        self.hand = hand
        #self.yard = yard
        
    def draw(self,deck):
        (cast(Hand, self.hand)).cards.append(cast(Deck, deck).cards.pop())
        
        #print((cast(Hand, self.hand)).cards)
        #print(cast(Deck, deck).cards)
        
    def turn(self):
        #compute moves
        pass
    
    def cleanUp(self) -> bool:
        return True