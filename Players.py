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
    
class HueristicHighCard(Player):
    def __init__(self, name):
        super().__init__(name)
        
    def getAction(self, state, mask):
        actOut = 0
        for x in mask:
            if x in range(1,53):
                actOut = x
        return actOut
            
    
class Agent(Player):
    def __init__(self, name, model, *args):
        super().__init__(name)
        #Set up policy and target model
        self.model = model
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
        self.memory = ReplayMemory(50000)
        
        #Using Adam optimization
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps = 1e-8, weight_decay=0)
        
        #using Huber Loss
        self.criterion = torch.nn.HuberLoss()
    
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
                        actout[x] = float('-inf')
                return actout.argmax().item()
        else:
            return torch.tensor([[random.choice(mask)]], dtype=torch.long).item()
        
    def get_state(self, ob):
        state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
        stateT = torch.from_numpy(np.array(state)).float()
        return stateT
    
    def optimize(self):
        if len(self.memory) < self.batchSize: return
        
        transitions = self.memory.sample(self.batchSize)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        
        state_action_values = self.policy(state_batch).gather(1, action_batch.unsqueeze(0))
        
        next_state_values = torch.zeros(self.batchSize)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values # type: ignore
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + torch.tensor(reward_batch)

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
    
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

        