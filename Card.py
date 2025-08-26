#defines card functionality based on number and stores the suit

class Card:
    number = 0
    suit = 0
    #Actual numbers are 1 to 13 and suits are 1 to 4
    
    def __init__(num, su):
        number = num
        suit = su