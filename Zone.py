#Defines the Zone type while also defining the functionality of the individual zone subclasses.
import random

from Card import Card




class Zone():
    
    def __init__(self, privacy, owner, cards):
        self.cards = []
        self.privacy = privacy
        self.owner = owner
        
    #privacy = 0 means no player knows, 1 means the owner knows, and 2 means both players know.
    #owner, 0 means no ownership, 1 means player 1, 2 means player 2
    
    def __str__(self) -> str:
        result = ""
        for x in self.cards:
            result += x.__str__()
            result += " | "
        return result
    
class Deck(Zone):
    def __init__(self):
        super().__init__(0, 0, [])
        #Once cards exist initialize the deck and shuffle here.
        #print(self.privacy)
        for x in range(1, 14):
            for y in range(1, 5):
                self.cards.append(Card(x,y))
            pass
            
        random.shuffle(self.cards)
        
class Hand(Zone):
    def __init__(self, owner):
        super().__init__(1, owner, [])
        



        

    

    