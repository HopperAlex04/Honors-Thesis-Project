import numpy as np
import torch

from GameEnvironment import CuttleEnvironment
from Players import Agent, Player


class WinRewardTraining:
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

    def trainLoop(self, episodes=1):
        # Data for tracking performace
        self.total_steps = 0
        p1wins = 0
        p2wins = 0

        # Initializing environment
        self.env = CuttleEnvironment()

        for episode in range(episodes):
            # Resets the deck and zones, then fills player hands.
            self.env.reset()
            turn = 0
            p1score = 0
            p2score = 0

            draw_counter = 0
            # Game Loop
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

                print(
                    f"{self.player1.name} Score: {p1score}, {self.player2.name} Score: {p2score}, Turns: {turn}"
                )

                # get an action from 'player'
                mask = self.env.generateActionMask()
                ob = self.env.get_obs()

                # Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1_act = self.player1.getAction(
                        ob, mask, self.env.actions, turn
                    )
                else:
                    self.p1_act = self.player1.getAction(ob, mask)

                if self.p1_act == 0:
                    draw_counter += 1
                else:
                    draw_counter = 0

                self.p1_state = self.get_state(ob)
                ob, p1score, terminated, truncated = self.env.step(self.p1_act)

                self.p2_next = self.get_state(ob)

                truncated = draw_counter >= 6

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
                    self.player2.memory.push(
                        torch.tensor(self.p2_state),
                        torch.tensor([self.p2_act]),
                        torch.tensor(self.p2_next),
                        torch.tensor([0]),
                    )

                self.env.passControl()

                # get an action from the 'dealer'
                mask = self.env.generateActionMask()
                ob = self.env.get_obs()

                # Gets an action, needs more parameters for agent actions
                if isinstance(self.player2, Agent):
                    self.p2_act = self.player2.getAction(
                        ob, mask, self.env.actions, turn
                    )
                else:
                    self.p2_act = self.player2.getAction(ob, mask)

                if self.p2_act == 0:
                    draw_counter += 1
                else:
                    draw_counter = 0

                self.p2_state = self.get_state(ob)

                ob, p2score, terminated, truncated = self.env.step(self.p2_act)

                self.p1_next = self.get_state(ob)

                truncated = draw_counter >= 6

                if terminated:
                    p2wins += 1
                    if isinstance(self.player1, Agent):
                        self.p2Win()
                    break
                elif truncated:
                    if isinstance(self.player1, Agent):
                        self.draw()
                    break
                elif isinstance(self.player1, Agent):
                    self.player1.memory.push(
                        self.p1_state,
                        torch.tensor([self.p1_act]),
                        self.p1_next,
                        torch.tensor([0]),
                    )

                self.env.passControl()

            print(
                f"Episode {episode}| {self.player1.name} WR: {p1wins/(episode + 1)} | {self.player2.name} WR {p2wins/(episode + 1)} | Average Turns: {self.total_steps/(episode + 1)}"
            )
            if isinstance(self.player1, Agent):
                self.player1.optimize()
            if isinstance(self.player2, Agent) and self.player1 != self.player2:
                self.player2.optimize()

    def validLoop(self, p1, new_player, logging=False, episodes=1):
        # Allows the ability to validate against other opponents
        self.player1 = p1
        self.player2 = new_player
        total_turns = 0
        # Data for tracking performace
        p1wins = 0
        p2wins = 0

        # Initializing environment
        self.env = CuttleEnvironment()

        p1log = [0] * self.env.actions
        p2log = [0] * self.env.actions
        for episode in range(episodes):
            # Resets the deck and zones, then fills player hands.
            self.env.reset()
            turn = 0
            p1score = 0
            p2score = 0
            draw_counter = 0
            # Game Loop
            terminated = False

            while not terminated:
                turn += 1
                total_turns += 1

                # get an action from 'player'
                mask = self.env.generateActionMask()
                ob = self.env.get_obs()

                # Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p1_act = self.player1.getAction(
                        ob, mask, self.env.actions, turn
                    )
                else:
                    self.p1_act = self.player1.getAction(ob, mask)

                if logging:
                    p1log[self.p1_act] += 1

                if len(mask) == 1:
                    draw_counter += 1
                else:
                    draw_counter = 0

                ob, p1score, terminated, truncated = self.env.step(self.p1_act)

                if not truncated:
                    truncated = draw_counter >= 6

                if terminated and turn > 1:
                    p1wins += 1
                    break
                elif truncated and turn > 1:
                    break

                self.env.passControl()

                # get an action from the 'dealer'
                mask = self.env.generateActionMask()
                ob = self.env.get_obs()

                # Gets an action, needs more parameters for agent actions
                if isinstance(self.player1, Agent):
                    self.p2_act = self.player1.getAction(
                        ob, mask, self.env.actions, turn
                    )
                else:
                    self.p2_act = self.player1.getAction(ob, mask)

                if logging:
                    p2log[self.p2_act] += 1

                if len(mask) == 1:
                    draw_counter += 1
                else:
                    draw_counter = 0

                ob, p2score, terminated, truncated = self.env.step(self.p2_act)

                if not truncated:
                    truncated = draw_counter >= 6

                if terminated:
                    p2wins += 1
                    break
                elif truncated:
                    break

                self.env.passControl()

                print(
                    f"{self.player1.name} Score: {p1score}, {self.player2.name} Score: {p2score}, Turns: {turn}"
                )

            print(
                f"Episode {episode}| {self.player1.name} WR: {p1wins/(episode + 1)} | {self.player2.name} WR {p2wins/(episode + 1)} | Average Turns: {total_turns/(episode + 1)}"
            )
        p1wr = p1wins / (episode + 1)
        p2wr = p2wins / (episode + 1)

        if logging:
            self.writelog(p1log, p2log)

        return p1wr, p2wr

    def get_state(self, ob):
        state = np.concatenate(
            (
                ob["Current Zones"]["Hand"],
                ob["Current Zones"]["Field"],
                ob["Current Zones"]["Revealed"],
                ob["Off-Player Field"],
                ob["Off-Player Revealed"],
                ob["Deck"],
                ob["Scrap"],
            ),
            axis=0,
        )
        embed_stack = self.model.embedding(torch.tensor(ob["Stack"]))
        embed_effect = self.model.embedding(torch.tensor(ob["Effect-Shown"]))
        state_tensor = torch.from_numpy(np.array(state)).float()

        embed_stack = torch.flatten(embed_stack, end_dim=-1)
        embed_effect = torch.flatten(embed_effect, end_dim=-1)

        final = torch.cat([state_tensor, embed_stack, embed_effect])

        return final

    def p1Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(
                self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([1])
            )
        if isinstance(self.player2, Agent):
            self.player2.memory.push(
                self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([-1])
            )

    def p2Win(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(
                self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([-1])
            )
        if isinstance(self.player2, Agent):
            self.player2.memory.push(
                self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([1])
            )

    def draw(self):
        if isinstance(self.player1, Agent):
            self.player1.memory.push(
                self.p1_state, torch.tensor([self.p1_act]), None, torch.tensor([0])
            )
        if isinstance(self.player2, Agent):
            self.player2.memory.push(
                self.p2_state, torch.tensor([self.p2_act]), None, torch.tensor([0])
            )

    def writelog(self, p1log, p2log):
        with open("p1log.txt", "w", encoding="utf-8") as f:
            current_type = None
            total_taken = 0
            for move, count in enumerate(p1log):
                # Figure out the type of move if it is different from current, switch current
                move_type = self.env.action_to_move[move][0]
                if move_type != current_type:
                    f.write(f"{current_type}\n{total_taken}\n")
                    current_type = move_type
                    total_taken = 0
                total_taken += count
            f.write(f"{current_type}\n{total_taken}\n")

        with open("p2log.txt", "w", encoding="utf-8") as f:
            current_type = None
            total_taken = 0
            for move, count in enumerate(p2log):
                # Figure out the type of move if it is different from current, switch current
                move_type = self.env.action_to_move[move][0]
                if move_type != current_type:
                    f.write(f"{current_type}\n{total_taken}\n")
                    current_type = move_type
                    total_taken = 0
                total_taken += count
