from Training import WinRewardTraining
from Players import Randomized

p1 = Randomized("Player")
p2 = Randomized("Dealer")

t1 = WinRewardTraining(p1, p2)

t1.train()