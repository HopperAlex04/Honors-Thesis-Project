import random
from Moves import Draw, Move, ScorePlay, ScuttlePlay
from Person import Player
from Zone import Hand



class Manual(Player):
    def __init__(self, hand: Hand, name: str) -> None:
        super().__init__(hand, name)
        hand.owner = self
        
    def getInput(self, zones:list) -> Move:
        finalMove: Move
        inOne = input("Select a card to play, or draw")
        handSize = range(self.hand.cards.__len__())
        if inOne == "draw": 
            finalMove = Draw(self.hand, zones[5]) 
        elif int(inOne) in handSize:
            selected = self.hand.cards[int(inOne)]
            print(selected)
            inTwo = input("scuttle or score?")
            if inTwo == "scuttle":
                inThree = input("target?")
                try:
                    targ = zones[3].cards[int(inThree)]
                    finalMove = ScuttlePlay(selected, targ, self.hand, zones[3], zones[4])
                except IndexError:
                    print("invalid target")
            elif inTwo == "score":
                finalMove = ScorePlay(selected, self.hand, zones[1])
        return finalMove    
    
    def turn(self, zones:list):
        super().turn(zones)
        valid = False
        
        while not valid:
            finalMove = self.getInput(zones)
            print(finalMove)
            for x in self.moves:
                #print(finalMove.__eq__(x))
                valid = finalMove.__eq__(x)
                if valid: break
            if not valid: print("invalid move")
            
        finalMove.execute()
            




class Randomized(Player):
    def __init__(self, hand, name: str) -> None:
        super().__init__(hand, name)
        hand.owner = self
    
    def turn(self, zones: list):
        super().turn(zones)
        if self.moves:
            select = random.randint(0, self.moves.__len__() - 1)
            self.moves[select].execute()