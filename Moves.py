from Card import Card
from Zone import Hand, Zone


class ScorePlay():
    
    def __init__(self, card: Card, hand: Hand, field: Zone):
        self.card = card
        self.field = field
        self.hand = hand
        
        
    #Validity of a maove is checked by referencing computed moves, so no need to check here.
    #score changes are done in a "cleanup step" based on the entire field.
    def execute(self):
        #print(self.card)
        #print(self.hand.cards)
        self.field.cards.append(self.hand.cards.pop(self.hand.cards.index(self.card)))
        #print(self.hand.cards)
        #print(self.field.cards)