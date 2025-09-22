from Cuttle import Cuttle
from GradientTesting import Agent, CuttleEnvironment
from Input import Randomized, Manual
from Person import Player
from Zone import Hand


#instance testing
count = 0
indices = []
indices.append(count)
for x in range(52):
    count += 1
    indices.append(count)
    
for x in range(52):
    indices.append(count)
    for y in range(x + 1):
        count += 1
    
print(indices)




