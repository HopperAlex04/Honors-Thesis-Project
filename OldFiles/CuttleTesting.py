from Cuttle import Cuttle
from Input import Randomized
from Zone import Hand

p = Randomized(Hand(0), "player")
d = Randomized(Hand(0), "dealer")

game = Cuttle(p, d)

game.gameStart()