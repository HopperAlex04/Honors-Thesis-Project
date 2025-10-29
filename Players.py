from abc import ABC, abstractmethod
from collections import deque, namedtuple
import math
import random

import numpy as np
import torch


class Player(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def getAction(self, state, mask):
        return 0
    
class Randomized(Player):
    def __init__(self, name, seed = 0):
        super().__init__(name)
        if seed != 0: random.seed(seed)
        
    def getAction(self, state, mask):
        return mask[random.randint(0, mask.__len__() - 1)]
    
class Agent(Player):
    def __init__(self, name, model, *args):
        super().__init__(name)
        #Set up policy and target model
        self.policy = model
        self.target = model
        self.target.load_state_dict(self.policy.state_dict())
        
        #Training parameters
        self.batchSize = args[0]
        self.gamma = args[1]
        self.epsStart = args[2]
        self.epsEnd = args[3]
        self.epsDecay = args[4]
        self.tau = args[5]
        self.lr = args[6]
        
        #Replay Memory
        self.memory = ReplayMemory(10000)
    
    def getAction(self, ob, mask, actions, steps_done):
        sample = random.random()
        eps_threshold = self.epsEnd + (self.epsStart - self.epsEnd) * \
            math.exp(-1. * steps_done / self.epsDecay)

        state = self.get_state(ob)
        
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                actout = self.policy(state)
                for x in range(actions):
                    if x not in mask:
                        actout[0, x] = float('-inf')
                return actout.max(1).indices.view(1,1).item()
        else:
            return torch.tensor([[random.choice(mask)]], dtype=torch.long).item()
        
    def get_state(self, ob):
        state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
        stateT = torch.from_numpy(np.array([state])).float()
        return stateT
    
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        print(f"{args[1]} {args[3]}")
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

        