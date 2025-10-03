from collections import deque, namedtuple
import random
from Person import Player
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np

#from Training import ReplayMemory # type: ignore

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQNAgent(Player):
    
    BATCH_SIZE = 128
    GAMMA = 0.5
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-2
    
    def __init__(self, hand, name, env, model, memory, device, data):
        super().__init__(hand, name)
        hand.owner = self
        self.env = env
        self.device =  device# type: ignore
        #print(f"Using {self.device} device")

        self.data = data
        
        self.model = model
        # Get number of actions from gym action space
        n_actions = 1379
        # Get the number of state observations
        #state, info = env.reset()
        n_observations = 52 * 4

        self.policy_net = DQNOne(n_observations, n_actions).to(self.device)
        self.target_net = DQNOne(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=DQNAgent.LR, amsgrad=True)
        self.memory = memory

        steps_done = 0
        #print(self.model)
        
    def turn(self, zones):
        super().turn(zones)
        
        self.env.envLoad(zones)
        valid = False
        
        #obs = self.env.get_obs
        state = np.concatenate((self.env.agentHandarr, self.env.agentFieldarr, self.env.oppFieldarr, self.env.scrapPilearr), axis = 0)
                
        stateT = torch.from_numpy(np.array([state])).float().to(device=self.device)
        #print(stateT)
        logits = self.model(stateT)
        
        mask = self.generateMask(zones)
        
        for x in range(1379):
            if x not in mask:
                logits[0, x] = float('-inf')
        
        
        probs = torch.softmax(logits, dim = 1)
        
        while not valid:
            pred_probab = nn.Softmax(dim=1)(probs)
            y_pred = pred_probab.argmax(1)
            mv = self.env.convertToMove(y_pred.item(), zones)
            #print(y_pred.item())
            for x in self.moves:
                valid = x.__eq__(mv)
                if valid: break
        
        
            
        ob, reward = self.env.step(y_pred.item(), zones)
        
        self.env.envLoad(zones)
        next_state = np.concatenate((self.env.agentHandarr, self.env.agentFieldarr, self.env.oppFieldarr, self.env.scrapPilearr), axis = 0)
        next_stateT = torch.from_numpy(np.array([next_state])).to(device=self.device)
        self.memory.push(stateT, y_pred, next_stateT, reward)

        self.updateModel()
        
        #print(ob, reward)
    
    def draw(self,deck):
        drawn = super().draw(deck)
        for x in self.hand.cards:
            self.env.agentHandarr[self.env.getIndex(drawn)]
            
    def generateMask(self, zones):
        computedMoves = []               
        for x in range(1379):
            computedMoves.append(self.env.convertToMove(x, zones))
        
        indexes = []

        for x in self.moves:
            indexes.append(computedMoves.index(x))
            
        return indexes
    
    def updateModel(self):
        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*DQNAgent.TAU + target_net_state_dict[key]*(1-DQNAgent.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        if self.score >= 21:
            self.data.append(1)
       
    def optimize_model(self):
        if len(self.memory) < DQNAgent.BATCH_SIZE:
            return
        transitions = self.memory.sample(DQNAgent.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state).to(self.device) 
        action_batch = torch.cat(batch.action).to(self.device) 
        reward_batch = torch.cat([torch.tensor(batch.reward)]).to(self.device) 
        next_batch = torch.cat(batch.next_state).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #print(self.policy_net(state_batch).shape)
        #print(action_batch.unsqueeze(1).shape)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(DQNAgent.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1).values
        # Compute the expected Q values
        #print(reward_batch.device)
        #print(reward_batch)
        expected_state_action_values = (next_state_values * DQNAgent.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    
class DQNOne(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, 1517),
            nn.ReLU(),
            nn.Linear(1517, 758),
            nn.ReLU(),
            nn.Linear(758, 380),
            nn.ReLU(),
            nn.Linear(380, 190),
            nn.ReLU(),
            nn.Linear(190, n_actions),
        )

    def forward(self, x):

        x = self.flatten(x)
        #print(x)
        logits = self.linear_relu_stack(x)
        return logits
    
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