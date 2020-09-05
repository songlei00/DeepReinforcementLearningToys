import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F 


def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)


class DeterministicPolicy(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(DeterministicPolicy, self).__init__()
        # TODO: add BN layer
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
            nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, state):
        return self.net(state)


class QFunction(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(QFunction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )
        self.apply(weight_init)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)


class ClippedDoubleQFunction(nn.Module):
    """Clipped double Q to deal with overestimation"""

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(ClippedDoubleQFunction, self).__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x1 = self.q1_net(x)
        x2 = self.q2_net(x)

        return x1, x2

    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        x1 = self.q1_net(x)
        return x1