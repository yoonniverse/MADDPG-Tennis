import torch
import torch.nn as nn


def init_head(module, bound):
    module.weight.data.uniform_(-bound, bound)
    module.bias.data.uniform_(-bound, bound)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=300):
        """
        :param state_size: (int)
        :param action_size: (int)
        :param hidden_size: (int)
        """
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, action_size)
        init_head(self.head, 3e-3)

    def forward(self, state):
        x = self.fc(state)
        action = self.head(x)
        return torch.tanh(action)  # action in [-1, 1]


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=300):
        """
        :param state_size: (int)
        :param action_size: (int)
        :param hidden_size: (int)
        """
        super(Critic, self).__init__()
        self.fc_state = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        self.fc_state_action = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, 1)
        init_head(self.head, 3e-3)

    def forward(self, state, action):
        s = self.fc_state(state)
        x = torch.cat((s, action), dim=-1)
        x = self.fc_state_action(x)
        q = self.head(x)
        return q
