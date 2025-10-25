from GameEnvironment import CuttleEnvironment
from Players import Player, Randomized

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
                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env._get_obs()
                action = 0
                action = self.player1.getAction(ob, mask)
                ob, p1Score, terminated = env.step(action)
                if terminated:
                    if p1Score >= 21: p1Wins += 1
                    break
                
                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env._get_obs()
                action = 0
                action = self.player2.getAction(ob, mask)
                ob, p2Score, terminated = env.step(action)
                if terminated:
                    if p2Score >= 21: p2Wins += 1
                    break
            print(f"{self.player1.name} Score: {p1Score}, {self.player2.name} Score: {p2Score}, Turns: {turn}")
        print(f"Episode End| {self.player1.name} WR: {p1Wins/(episode + 1)} | {self.player2.name} WR {p2Wins/(episode + 1)} | Average Turns: {total_turns/(episode + 1)}")
                
                
                
                
                
                
                
                

            
