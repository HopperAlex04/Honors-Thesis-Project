from typing import cast
from Zone import Deck, Hand


class Player():
    def __init__(self, hand):
        self.hand = hand
        #self.yard = yard
        
    def draw(self,deck):
        (cast(Hand, self.hand)).cards.append(cast(Deck, deck).cards.pop())
        
        print((cast(Hand, self.hand)).cards)
        print(cast(Deck, deck).cards)

        
d1 = Deck()
h1 = Hand(1)


p1 = Player(h1)

p1.draw(d1)
p1.draw(d1)
p1.draw(d1)