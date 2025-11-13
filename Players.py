import math
import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple

import numpy as np
import torch


class Player(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def getAction(self, state, mask):
        return 0


class Randomized(Player):
    def __init__(self, name, seed=None):
        super().__init__(name)
        random.seed(seed)

    def getAction(self, state, mask) -> int:
        return random.choice(mask)


class HueristicHighCard(Player):

    def getAction(self, state, mask) -> int:
        act_out = 0
        for x in mask:
            if x in range(1, 53):
                act_out = x
        return act_out


class Agent(Player):
    def __init__(self, name, model, *args):
        super().__init__(name)
        # Set up policy model

        self.model = model

        self.policy = model

        # Training parameters
        self.batch_size = args[0]
        self.gamma = args[1]
        self.eps_start = args[2]
        self.eps_end = args[3]
        self.eps_decay = args[4]
        self.tau = args[5]
        self.lr = args[6]

        # Replay Memory
        self.memory = ReplayMemory(50000)

        # Using Adam optimization
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )

        # using Huber Loss
        self.criterion = torch.nn.SmoothL1Loss()

    def getAction(self, ob, mask, actions, steps__done) -> int:
        sample = random.random()
        # As training continues, when getAction is called it becomes more likely to return an action from the model
        eps__threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * steps__done / self.eps_decay
        )
        # ob should be of the form [dict, dict, zone, zone] so the dicts need to be broken down by get_state()
        state = self.get_state(ob)

        if sample > eps__threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                act_out = self.policy(state)
                for x in range(actions):
                    if x not in mask:
                        act_out[x] = float("-inf")
                return act_out.argmax().item()
        else:
            return random.choice(mask)

    # States are dim 1 tensors (think [1, 0, 0, 1, 0, 1])
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

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        # The zip function: zip(iterator1, iterator2...) returns a zip object, which is an iterator of the form [(iterator1[0], iterator2[0]...]),...]

        # zip(*iterable) will use all of the elements of that iterable as arguements. zip(*transitions) is the collection of all the transitions such that
        #   (assuming array structure) each row corresponds to all of that part of the transition. ex. zip(*transitions)[0] would be all of the states, in order.

        # Transition(*zip(*transitions)) is the final step, which makes this collection able to be accesed as so: batch.field, where field is one of the namedTuple's names
        # defined as Transition(transition, (state, action, next_state, reward))
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # print(torch.stack(action_batch))
        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy(non_final_next_states).max(1).values  # type: ignore
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)

        self.optimizer.step()


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
