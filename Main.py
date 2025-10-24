#from EnvTests import drawTest, generateCardsTest, initTest, maskTest, resetTest, scoreTest, scuttleTest
#from NetworkTesting import getActionTest
from Training import randomLoop, winReward01
import torch

# initTest()
# input("Complete")

# generateCardsTest()
# input("Complete")

# resetTest()
# input("Complete")

# scoreTest()
# input("Complete")

# drawTest()
# input("Complete")

# scuttleTest()
# input("Complete")

# maskTest()
# input("Complete")

# randomLoop(10)
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
# print(f"Using {device} device")

#getActionTest()

winReward01(200)