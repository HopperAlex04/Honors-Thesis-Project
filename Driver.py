from Cuttle import Cuttle
from Person import Player
from Zone import Hand


game = Cuttle(Player(Hand(0)), Player(Hand(0)))

game.gameStart()


