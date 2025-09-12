import random
from Moves import Draw, Move, ScorePlay, ScuttlePlay
from Person import Player
from Zone import Hand

import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import torch.distributions as distributions # type: ignore
import numpy as np # type: ignore
import gymnasium as gym # type: ignore


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns

def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    observation, info = env.reset()

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)

    return episode_return, stepwise_returns, log_prob_actions

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main(): 
    MAX_EPOCHS = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = env.observation_space.shape[0] #test altering this
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n #Test altering this
    DROPOUT = 0.5

    episode_returns = []

    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

    LEARNING_RATE = 0.01
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)

        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')

        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break


env = gym.make('CartPole-v1') #This is the target of alteration I think
print(env.action_space.n)
#main()

class Manual(Player):
    def __init__(self, hand: Hand, name: str) -> None:
        super().__init__(hand, name)
        
    def getInput(self, zones:list) -> Move:
        finalMove: Move
        inOne = input("Select a card to play, or draw")
        handSize = range(self.hand.cards.__len__())
        if inOne == "draw": 
            finalMove = Draw(self.hand, zones[5]) 
        elif int(inOne) in handSize:
            selected = self.hand.cards[int(inOne)]
            print(selected)
            inTwo = input("scuttle or score?")
            if inTwo == "scuttle":
                inThree = input("target?")
                try:
                    targ = zones[3].cards[int(inThree)]
                    finalMove = ScuttlePlay(selected, targ, self.hand, zones[3], zones[4])
                except IndexError:
                    print("invalid target")
            elif inTwo == "score":
                finalMove = ScorePlay(selected, self.hand, zones[1])
        return finalMove    
    
    def turn(self, zones:list):
        super().turn(zones)
        valid = False
        
        while not valid:
            finalMove = self.getInput(zones)
            print(finalMove)
            for x in self.moves:
                #print(finalMove.__eq__(x))
                valid = finalMove.__eq__(x)
                if valid: break
            if not valid: print("invalid move")
            
        finalMove.execute()
            




class Randomized(Player):
    def __init__(self, hand: Hand, name: str) -> None:
        super().__init__(hand, name)
    
    def turn(self, zones: list):
        super().turn(zones)
        select = random.randint(0, self.moves.__len__() - 1)
        self.moves[select].execute()