#Defines the Zone type while also defining the functionality of the individual zone subclasses.
class Zone(ABC):
    cards = str[]
    privacy = 0
    owner = 0
    #privacy = 0 means no player knows, 1 means the owner knows, and 2 means both players know.
    #owner, 0 means no ownership, 1 means player 1, 2 means player 2
    
class Deck(Zone):
    def __init__(o):
        owner = o
        privacy = 1
        #Once cards exist initialize the deck and shuffle here.
        

    

    