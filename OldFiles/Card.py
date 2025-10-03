#defines card functionality based on number and stores the suit

class Card:
    #Actual numbers are 1 to 13 and suits are 1 to 4
    
    def __init__(self, num, su):
        self.number = num
        self.suit = su
        
    def __str__(self) -> str:
        return str(self.number) + " " + str(self.suit)