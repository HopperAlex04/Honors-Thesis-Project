#Defines the Zone type while also defining the functionality of the individual zone subclasses.
import random




class Zone():
    
    def __init__(self, privacy, owner, cards):
        self.cards = []
        self.privacy = privacy
        self.owner = owner
        
    #privacy = 0 means no player knows, 1 means the owner knows, and 2 means both players know.
    #owner, 0 means no ownership, 1 means player 1, 2 means player 2
    
class Deck(Zone):
    def __init__(self):
        super().__init__(0, 0, [])
        #Once cards exist initialize the deck and shuffle here.
        #print(self.privacy)
        for x in range(0, 52):
            self.cards.append(x)
        random.shuffle(self.cards)
        
class Hand(Zone):
    def __init__(self, owner):
        super().__init__(1, owner, [])
        



        

    

    