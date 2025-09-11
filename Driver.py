from Cuttle import Cuttle
from Input import Manual
from Person import Player
from Zone import Hand


#instance testing
p1 = Manual(Hand(0), "dealer")
p2 = Manual(Hand(0), "player")

game = Cuttle(p1, p2)

game.gameStart()

#scale testing

#for x in range (10000):
#    game = Cuttle(Player(Hand(0), "dealer"), Player(Hand(0), "player"))
#    game.gameStart()


