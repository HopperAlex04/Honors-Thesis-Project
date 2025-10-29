import numpy as np
import torch
from GameEnvironment import CuttleEnvironment
from Players import Agent, Player, Randomized

class WinRewardTraining():
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
    
    
    def train(self, episodes = 1):
        #Data for tracking performace
        total_turns = 0
        p1Wins = 0
        p2Wins = 0
        
        #Initializing environment
        env = CuttleEnvironment()
        
        
        for episode in range(episodes):
            #Resets the deck and zones, then fills player hands.
            env.reset()
            turn = 0
            p1Score = 0
            p2Score = 0
            #Game Loop
            terminated = False
            while not terminated:
                turn += 1
                total_turns += 1
                
                p1State = None
                p1Act = 0
                p1Next = None
                
                p2State = None
                p2Act = 0
                p2Next = None
                
                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env._get_obs()
                action = 0
                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    p1Act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    p1Act = self.player1.getAction(ob, mask)
                    
                p1State = self.get_state(ob)
                
                ob, p1Score, terminated, truncated = env.step(p1Act)
                
                p2Next = self.get_state(ob)
                
                if terminated and turn > 1:
                    p1Wins += 1
                    if isinstance(self.player1, Agent):
                        self.player1.memory.push(p1State, p1Act, None, 1)
                    if isinstance(self.player2, Agent):
                        self.player2.memory.push(p2State, p2Act, None, -1)
                    break
                elif truncated and turn > 1:
                    if isinstance(self.player1, Agent):
                        self.player1.memory.push(p1State, p1Act, None, 0)
                    if isinstance(self.player2, Agent):
                        self.player2.memory.push(p2State, p2Act, None, 0)
                    break
                elif isinstance(self.player2, Agent) and turn > 1: 
                    self.player2.memory.push(p2State, p2Act, p2Next, 0)      
                
                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env._get_obs()
                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    p2Act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    p2Act = self.player1.getAction(ob, mask)
                    
                p2State = self.get_state(ob)
                
                ob, p2Score, terminated, truncated = env.step(p2Act)
                
                p1Next = self.get_state(ob)
                
                if terminated:
                    p2Wins += 1
                    if isinstance(self.player1, Agent):
                        self.player1.memory.push(p1State, p1Act, None, -1)
                    if isinstance(self.player2, Agent):
                        self.player2.memory.push(p2State, p2Act, p2Next, 1)
                    break
                elif truncated:
                    if isinstance(self.player1, Agent):
                        self.player1.memory.push(p1State, p1Act, None, 0)
                    if isinstance(self.player2, Agent):
                        self.player2.memory.push(p2State, p2Act, p2Next, 0)
                    break
                elif isinstance(self.player1, Agent): 
                    self.player1.memory.push(p1State, p1Act, p1Next, 0)
                    
                    
            print(f"{self.player1.name} Score: {p1Score}, {self.player2.name} Score: {p2Score}, Turns: {turn}")
            
        print(f"Episode End| {self.player1.name} WR: {p1Wins/(episode + 1)} | {self.player2.name} WR {p2Wins/(episode + 1)} | Average Turns: {total_turns/(episode + 1)}")
        
        
        
    def get_state(self, ob):
        state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
        stateT = torch.from_numpy(np.array([state])).float()
        return stateT
                
                
                
                
                
                
                
                

            
