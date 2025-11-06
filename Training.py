import numpy as np
import torch
from GameEnvironment import CuttleEnvironment
from Players import Agent, Player

class WinRewardTraining():
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
        self.total_steps = 0
        self.p1_state = None
        self.p1_act = 0
        self.p1_next = None
        self.p2_state = None
        self.p2_act = 0
        self.p2_next = None


    def trainLoop(self, episodes = 1):
        #Data for tracking performace
        self.total_steps = 0
        p1wins = 0
        p2wins = 0

        #Initializing environment
        env = CuttleEnvironment()


        for episode in range(episodes):
            #Resets the deck and zones, then fills player hands.
            env.reset()
            turn = 0
            p1score = 0
            p2score = 0

            draw_counter = 0
            #Game Loop
            terminated = False

            self.p1_state = None
            self.p1_act = 0
            self.p1_next = None

            self.p2_state = None
            self.p2_act = 0
            self.p2_next = None


            while not terminated:
                turn += 1
                self.total_steps += 1

                print(f"{self.player1.name} Score: {p1score}, {self.player2.name} Score: {p2score}, Turns: {turn}")

                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env.get_obs()

                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1_act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p1_act = self.player1.getAction(ob, mask)

                if self.p1_act == 0: draw_counter += 1
                else: draw_counter = 0

                self.p1_state = self.get_state(ob)
                ob, p1score, terminated, truncated = env.step(self.p1_act)

                self.p2_next = self.get_state(ob)

                truncated = (draw_counter >= 6)

                if terminated and turn > 1:
                    p1wins += 1
                    if isinstance(self.player1, Agent):
                        self.p1Win()
                    break
                elif truncated and turn > 1:
                    if isinstance(self.player1, Agent):
                        self.draw()
                    break
                elif isinstance(self.player2, Agent) and turn > 1:
                    self.player2.memory.push(torch.tensor(self.p2_state), torch.tensor([self.p2_act]), torch.tensor(self.p2_next), torch.tensor([0]))

                env.passControl()

                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env.get_obs()

                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player2, Agent):
                    self.p2_act = self.player2.getAction(ob, mask, env.actions, turn)
                else:
                    self.p2_act = self.player2.getAction(ob, mask)

                if self.p2_act == 0: draw_counter += 1
                else: draw_counter = 0

                self.p2_state = self.get_state(ob)

                ob, p2score, terminated, truncated = env.step(self.p2_act)

                self.p1_next = self.get_state(ob)

                truncated = (draw_counter >= 6)

                if terminated:
                    p2wins += 1
                    if isinstance(self.player1, Agent): self.p2Win()
                    break
                elif truncated:
                    if isinstance(self.player1, Agent): self.draw()
                    break
                elif isinstance(self.player1, Agent):
                    self.player1.memory.push(self.p1_state, torch.tensor([self.p1_act]), self.p1_next, torch.tensor([0]))

                env.passControl()




            print(f"Episode {episode}| {self.player1.name} WR: {p1wins/(episode + 1)} | {self.player2.name} WR {p2wins/(episode + 1)} | Average Turns: {self.total_steps/(episode + 1)}")
            if isinstance(self.player1, Agent): self.player1.optimize()
            if isinstance(self.player2, Agent) and self.player1 != self.player2: self.player2.optimize()

    def validLoop(self, p1, newPlayer, episodes = 1):
        #Allows the ability to validate against other opponents
        self.player1 = p1
        self.player2 = newPlayer
        totalTurns = 0
        #Data for tracking performace
        p1wins = 0
        p2wins = 0

        #Initializing environment
        env = CuttleEnvironment()


        for episode in range(episodes):
            #Resets the deck and zones, then fills player hands.
            env.reset()
            turn = 0
            p1score = 0
            p2score = 0
            #Game Loop
            terminated = False

            while not terminated:
                turn += 1
                totalTurns += 1

                #get an action from 'player'
                mask = env.generateActionMask()
                ob = env.get_obs()

                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1_act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p1_act = self.player1.getAction(ob, mask)

                if len(mask) == 1: draw_counter += 1
                else: draw_counter = 0

                ob, p1score, terminated, truncated = env.step(self.p1_act)

                if not truncated:
                    truncated = (draw_counter >= 6)

                if terminated and turn > 1:
                    p1wins += 1
                    break
                elif truncated and turn > 1:
                    break

                env.passControl()

                #get an action from the 'dealer'
                mask = env.generateActionMask()
                ob = env.get_obs()

                #Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p2_act = self.player1.getAction(ob, mask, env.actions, turn)
                else:
                    self.p2_act = self.player1.getAction(ob, mask)

                if len(mask) == 1: draw_counter += 1
                else: draw_counter = 0

                ob, p2score, terminated, truncated = env.step(self.p2_act)

                truncated = (draw_counter >= 6)

                if terminated:
                    p2wins += 1
                    break
                elif truncated:
                    break

                env.passControl()

                print(f"{self.player1.name} Score: {p1score}, {self.player2.name} Score: {p2score}, Turns: {turn}")

            print(f"Episode {episode}| {self.player1.name} WR: {p1wins/(episode + 1)} | {self.player2.name} WR {p2wins/(episode + 1)} | Average Turns: {totalTurns/(episode + 1)}")
        p1WR = p1wins/(episode + 1)
        p2WR = p2wins/(episode + 1)
        return p1WR, p2WR

    def get_state(self, ob):
        state = np.concatenate((ob["Current Zones"]["Hand"], ob["Current Zones"]["Field"], ob["Off-Player Zones"]["Hand"], ob["Deck"], ob["Scrap"]), axis = 0)
        stateT = torch.from_numpy(np.array(state)).float()
        return stateT

    def p1Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([1]))
        if isinstance(self.player2, Agent):
            self.player2.memory.push(self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([-1]))

    def p2Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([-1]))
        if isinstance(self.player2, Agent):
            self.player2.memory.push(self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([1]))

    def draw(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([0]))
        if isinstance(self.player2, Agent):
            self.player2.memory.push(self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([0]))












