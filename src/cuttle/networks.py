import numpy as np
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(
        self, obspace, embedding_size, actions: int, seq: nn.Sequential | None
    ):
        super().__init__()
        # self.flatten = nn.Flatten()
        if seq:
            self.linear_relu_stack = seq
        else:
            input_length = 0
            for item in obspace.values():
                if isinstance(item, dict):
                    for x in item.values():
                        input_length += len(x)
                if isinstance(item, np.ndarray):
                    input_length += len(item)
                elif isinstance(item, list):
                    for _ in item:
                        input_length += embedding_size
            self.embedding = nn.Embedding(
                54, embedding_size
            )  # 0 indicates nothing is there, 1-52 are cards
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_length, actions), nn.Tanh()
            )

    def forward(self, x):

        state = self.get_state(x)
        logits = self.linear_relu_stack(state)
        return logits

    def get_state(self, x):
        if isinstance(x, dict):
            state = np.concatenate(
                (
                    x["Current Zones"]["Hand"],
                    x["Current Zones"]["Field"],
                    x["Current Zones"]["Revealed"],
                    x["Off-Player Field"],
                    x["Off-Player Revealed"],
                    x["Deck"],
                    x["Scrap"],
                ),
                axis=0,
            )
            embed_stack = self.embedding(torch.tensor(x["Stack"]))
            embed_effect = self.embedding(torch.tensor(x["Effect-Shown"]))
            state_tensor = torch.from_numpy(np.array(state)).float()

            embed_stack = torch.flatten(embed_stack, end_dim=-1)
            embed_effect = torch.flatten(embed_effect, end_dim=-1)

            final = torch.cat([state_tensor, embed_stack, embed_effect])

            return final
        else:
            final_list = []
            for state in x:
                final_list.append(self.get_state(state))
            final_ten = torch.empty(378, 3503)
            if final_list:
                final_ten = torch.stack(final_list)

            return final_ten
