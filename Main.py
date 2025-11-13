import torch

import Players
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork
from Training import WinRewardTraining

input("Press enter to begin training:")

user_ended = False

env = CuttleEnvironment()
actions = env.actions
model = NeuralNetwork(env.observation_space, 2, actions, None)

# Eventually make these adjustable as well
BATCH_SIZE = 4096
GAMMA = 0.4
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
trainee = Players.Agent(
    "Agent01", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR
)
validation01 = Players.HueristicHighCard("HighCard")

t = WinRewardTraining(trainee, trainee)

while not user_ended:
    user_in = input("Enter 't' to train,'v' to validate, 's' to save, or q to quit")
    match user_in:
        case "t":
            t.trainLoop(500)
        case "v":
            t.validLoop(trainee, validation01, True, 500)
        case "s":
            model_name = input("Name the model")
            torch.save(trainee.model, f"models/{model_name}")
        case "q":
            user_ended = True
