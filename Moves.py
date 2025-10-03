from abc import ABC, abstractmethod
from Card import Card
from Zone import Deck, Hand, Zone


class Move(ABC):
    @abstractmethod
    def execute(self):
        pass

class ScorePlay(Move):
    
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
        print(self.hand.owner.name)
        print("scored")
        print(self.card)
        
    def __eq__(self, other):
        equals = True
        #print(other)
        if not isinstance(other, ScorePlay):
            #print("not a score")
            equals = False
        else:
            #print(self.field)
            #print(other.field)
            #print(self.card == other.card)
            #print(self.field == other.field)
            #print(self.hand == other.hand)
            #print()
            
            equals = (self.card == other.card) and (self.field == other.field) and (self.hand == other.hand)
            
        return equals
        
class ScuttlePlay(Move):
    def __init__(self, card:Card, target:Card, hand:Hand, field:Zone, scrap:Zone):
        self.card = card
        self.target = target
        self.field = field
        self.hand = hand
        self.scrap = scrap
        
    def execute(self):
        self.scrap.cards.append(self.hand.cards.pop(self.hand.cards.index(self.card)))
        self.scrap.cards.append(self.field.cards.pop(self.field.cards.index(self.target)))
        
        print(self.hand.owner.name)
        print("scuttled")
        print(self.target)
        print("with")
        print(self.card)
        
    def __eq__(self, other):
        equals = True
        if not isinstance(other, ScuttlePlay):
            equals = False
        else:
            equals = (self.card == other.card) and (self.field == other.field) and (self.hand == other.hand) and (self.target == other.target) and (self.scrap == other.scrap)
                
        return equals
        
class Draw(Move):
    def __init__(self, hand:Hand, deck:Deck):
        self.hand = hand
        self.deck = deck
    
    def execute(self):
        if self.deck:
            self.hand.cards.append(self.deck.cards.pop())
        print(self.hand.owner.name)
        print("drew")
        
    def __eq__(self, other):
        equals = True
        if not isinstance(other, Draw):
            equals = False
        else:
            equals = (self.hand == other.hand) and (self.deck == other.deck)
            
        return equals
    
class AceAction(Move):
    def __init__(self, card, hand:Hand, pField:Zone, oField:Zone, scrap:Zone):
        self.card = card
        self.hand = hand
        self.pField = pField
        self.oField = oField
        self.scrap = scrap
        
    def execute(self):
        self.scrap.cards.append(self.hand.cards.pop(self.hand.cards.index(self.card)))
        while self.pField.cards:
            self.scrap.cards.append(self.pField.cards.pop())
        while self.oField.cards:
            self.scrap.cards.append(self.oField.cards.pop())
        
        print(self.hand.owner.name)
        print("Board Wiped")
        print("with")
        print(self.card)
        
    def __eq__(self, other):
        equals = True
        if not isinstance(other, AceAction):
            equals = False
        else:
            equals = (self.card == other.card) and (self.pField == other.pField) and (self.hand == other.hand) and (self.oField == other.oField) and (self.scrap == other.scrap)
            
        return equals