#Treat the game as an arbitrary generator of states based on input actions, setting up a turn order as follows
from abc import ABC, abstractmethod
from itertools import count
import math
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import torch
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork, ReplayMemory, Transition, get_action, get_state


class Randomized():
    def __init__(self, name):
        self.name = name

    def getAction(self, validActions:list) -> int:
        return validActions[random.randint(0, validActions.__len__() - 1)]


#Loop using randomized action generation for testing
def randomLoop(episodes):
    env = CuttleEnvironment()

    p1 = Randomized("p1")
    for x in range(episodes):
        env.reset()
        
        terminated = False

        while not terminated:
            #Get an action from the player and do it
            validActions = env.generateActionMask()
            action = p1.getAction(validActions)
            ob, score, terminated = env.step(action)
            
            #ob, score, terminated, and truncated (when I implement it), would be given to a model
            
            #Example: p1.updateModel(ob, score, terminated, truncated)
            
            #This could probably be part of the step function
            env.render()
            env.passControl()
            
def winReward01(episodes = 0, b_size = 128, gam = 0.99, e_start = 0.9, e_end = 0.01, e_dec = 2500, tau = 0.005, lr = 3e-4):
    
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
    env = CuttleEnvironment()
    actions = env.actions
    p1 = Randomized("p1")
    
    
    BATCH_SIZE = b_size
    GAMMA = gam
    EPS_START = e_start
    EPS_END = e_end
    EPS_DECAY = e_dec
    TAU = tau
    LR = lr

    policy_net = NeuralNetwork(52 * 5, actions).to(device)
    target_net = NeuralNetwork(52 * 5, actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    global steps_done
    steps_done = 0
    
    #Think I need to add masking here
    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                actout = policy_net(state).to(device)
                for x in range(actions):
                    if x not in env.generateActionMask():
                        actout[0, x] = float('-inf')
                return actout.max(1).indices.view(1,1)
        else:
            return torch.tensor([[random.choice(env.generateActionMask())]], device=device, dtype=torch.long).to(device)
        
    episodeWins = []
    
    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episodeWins, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
                
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
    if episodes <= 0:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            episodes = 600
        else:
            episodes = 50
            
    for i_episode in range(episodes):
        # Initialize the environment and get its state
        start = time.time()
        env.reset()
        terminated = False
        while not terminated:
            state = get_state(env._get_obs())
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state)
            
            #Here I need to change stuff
            observation, score, terminated = env.step(action.item())
            env.render()
            #Basically make it two player with the below reward calcs being done after the random turn.
            if score >= 21: # type: ignore
                reward = 1 
                episodeWins.append(1)
                plot_durations()
            else: reward = 0
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                next_state = None
            else:
                env.passControl()
                observation, score, terminated = env.step(p1.getAction(env.generateActionMask()))
                if score >= 21 or terminated: # type: ignore
                    episodeWins.append(0)
                    plot_durations()
                    break 
                env.passControl()
                next_state = env._get_obs()

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
        
        end = time.time()
        print(f"Episode: {i_episode}")
        print(f"Duration: {end - start} seconds")

                

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

            
