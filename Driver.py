from Cuttle import Cuttle
from Person import Player
from Zone import Hand


#instance testing
game = Cuttle(Player(Hand(0), "dealer"), Player(Hand(0), "player"))

game.gameStart()

#scale testing

#for x in range (10000):
#    game = Cuttle(Player(Hand(0), "dealer"), Player(Hand(0), "player"))
#    game.gameStart()


