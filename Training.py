import numpy as np
import torch
from GameEnvironment import CuttleEnvironment
from Players import Agent, Player

class WinRewardTraining():
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
        self.total_steps = 0
    
    
    def trainLoop(self, episodes = 1):
        #Data for tracking performace
        self.total_steps = 0
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
            
            drawCounter = 0
            #Game Loop
            terminated = False
            
            self.p1State = None
            self.p1Act = 0
            self.p1Next = None
                
            self.p2State = None
            self.p2Act = 0
            self.p2Next = None
            
            
            while not terminated:
                turn += 1
                self.total_steps += 1
                
                print(f"{self.player1.name} Score: {p1Score}, {self.player2.name} Score: {p2Score}, Turns: {turn}")
                
                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env._get_obs()
                                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1Act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p1Act = self.player1.getAction(ob, mask)
                    
                if self.p1Act == 0: drawCounter += 1
                else: drawCounter = 0
                    
                self.p1State = self.get_state(ob)
                print(self.p1Act)
                ob, p1Score, terminated, truncated = env.step(self.p1Act)
                
                self.p2Next = self.get_state(ob)
                
                truncated = (drawCounter >= 6)
                
                if terminated and turn > 1:
                    p1Wins += 1
                    if isinstance(self.player1, Agent):
                        self.p1Win()
                    break
                elif truncated and turn > 1:
                    if isinstance(self.player1, Agent):
                        self.draw()
                    break
                elif isinstance(self.player2, Agent) and turn > 1: 
                    self.player2.memory.push(torch.tensor(self.p2State), torch.tensor(self.p2Act), torch.tensor(self.p2Next), 0)      
                
                env.passControl()
                
                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env._get_obs()
                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player2, Agent):
                    self.p2Act = self.player2.getAction(ob, mask, env.actions, turn)
                else:
                    self.p2Act = self.player2.getAction(ob, mask)
                    
                if self.p2Act == 0: drawCounter += 1
                else: drawCounter = 0
                    
                self.p2State = self.get_state(ob)
                
                ob, p2Score, terminated, truncated = env.step(self.p2Act)
                
                self.p1Next = self.get_state(ob)
                
                truncated = (drawCounter >= 6)
                
                if terminated:
                    p2Wins += 1
                    if isinstance(self.player1, Agent): self.p2Win()
                    break
                elif truncated:
                    if isinstance(self.player1, Agent): self.draw()
                    break
                elif isinstance(self.player1, Agent): 
                    self.player1.memory.push(torch.tensor(self.p1State), torch.tensor(self.p1Act), torch.tensor(self.p1Next), 0)
                    
                env.passControl()
                    
                    
            
            
            print(f"Episode {episode}| {self.player1.name} WR: {p1Wins/(episode + 1)} | {self.player2.name} WR {p2Wins/(episode + 1)} | Average Turns: {self.total_steps/(episode + 1)}")
            if isinstance(self.player1, Agent) and isinstance(self.player2, Agent):
                self.player1.optimize()
                if self.player1.model != self.player2.model:
                    self.player2.optimize()
                
    def validLoop(self, newPlayer, episodes = 1):
        #Allows the ability to validate against other opponents
        self.player2 = newPlayer
        totalTurns = 0
        #Data for tracking performace
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
                totalTurns += 1
                
                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env._get_obs()
                                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1Act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p1Act = self.player1.getAction(ob, mask)
                
                if len(mask) == 1: drawCounter += 1
                else: drawCounter = 0
                
                ob, p1Score, terminated, truncated = env.step(self.p1Act)
                
                truncated = (drawCounter >= 6)
                
                if terminated and turn > 1:
                    p1Wins += 1
                    break
                elif truncated and turn > 1:
                    break     
                
                env.passControl()
                
                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env._get_obs()
                
                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p2Act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p2Act = self.player1.getAction(ob, mask)
                
                if len(mask) == 1: drawCounter += 1
                else: drawCounter = 0
                
                ob, p2Score, terminated, truncated = env.step(self.p2Act)
                
                truncated = (drawCounter >= 6)                
                
                if terminated:
                    p2Wins += 1
                    break
                elif truncated:
                    break
                
                env.passControl()
                    
                print(f"{self.player1.name} Score: {p1Score}, {self.player2.name} Score: {p2Score}, Turns: {turn}")
            
            print(f"Episode {episode}| {self.player1.name} WR: {p1Wins/(episode + 1)} | {self.player2.name} WR {p2Wins/(episode + 1)} | Average Turns: {totalTurns/(episode + 1)}")
        
        
    def get_state(self, ob):
        state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
        stateT = torch.from_numpy(np.array(state)).float()
        return stateT
    
    def p1Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(torch.tensor(self.p1State), torch.tensor(self.p1Act), None, 1)
        if isinstance(self.player2, Agent):
            self.player2.memory.push(torch.tensor(self.p2State), torch.tensor(self.p2Act), None, -1)
            
    def p2Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(torch.tensor(self.p1State), torch.tensor(self.p1Act), None, -1)
        if isinstance(self.player2, Agent):
            self.player2.memory.push(torch.tensor(self.p2State), torch.tensor(self.p2Act), None , 1)
            
    def draw(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(torch.tensor(self.p1State), torch.tensor(self.p1Act), None, 0)
        if isinstance(self.player2, Agent):
            self.player2.memory.push(torch.tensor(self.p2State), torch.tensor(self.p2Act), None, 0)
        
        
                
                
                
                
                
                
                
                

            
