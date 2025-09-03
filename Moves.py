from Card import Card
from Zone import Hand, Zone


class scorePlay():
    card: Card
    field: Zone
    hand: Hand
    
    def __init__(self, card: Card, field: Zone, hand: Hand):
        self.card = card
        self.field = field
        self.hand = hand
        
    #method which when called on a move object, executes the move.