import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Softmax
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def weights_init_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.1)


class Actor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128):
        super(Actor, self).__init__()
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
            Softmax(dim=0),
        )
        self.apply(weights_init_)

    def forward(self, state):
        return self.net(state)

    def sample(self, state, is_test=False):
        pi = self.forward(state)
        dist = Categorical(pi)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class GaussianActor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128, action_scale=None):
        super(GaussianActor, self).__init__()
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.mean_linear = Linear(hidden_size, output_size)
        self.log_std_linear = Linear(hidden_size, output_size)

        self.apply(weights_init_)
        self.action_scale = 1 if action_scale is None else action_scale

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean_linear(x)) * self.action_scale
        log_std = self.log_std_linear(x)
        log_std = log_std.clamp(-20, 2)
        return mean, log_std

    def sample(self, state, is_test=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if is_test:
            action = mean
            log_prob = normal.log_prob(action)
        else:
            action = normal.sample()
            log_prob = normal.log_prob(action)
        action = action.clamp(-self.action_scale, self.action_scale)

        return action, log_prob

    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        log_prob = normal.log_prob(action)
        return log_prob


class SACGaussianActor(GaussianActor):

    def __init__(self, input_size, output_size, hidden_size=128, action_scale=None):
        GaussianActor.__init__(self, input_size, output_size, hidden_size, action_scale)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = log_std.clamp(-20, 2)
        return mean, log_std

    def sample(self, state, is_test=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if is_test:
            action = torch.tanh(mean) * self.action_scale
            y = torch.tanh(mean)
            log_prob = normal.log_prob(mean) - torch.log(1 - y.pow(2) + 1e-8)
        else:
            # reparameterization trick
            x = normal.rsample()
            y = torch.tanh(x)
            action = y * self.action_scale
            # Enforcing action bound
            # BUG: 应该对log_prob求和，输出的log_porb为向量
            log_prob = normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)

        return action, log_prob

    def get_log_prob(self, state, action):
        raise NotImplementedError("get_log_prob: not implemented")


class DeterministicActor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128, action_scale=None):
        super(DeterministicActor, self).__init__()
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
        )

        self.apply(weights_init_)
        self.action_scale = 1 if action_scale is None else action_scale

    def forward(self, state):
        x = self.net(state)
        return torch.tanh(x) * self.action_scale

    def sample(self, state):
        return self.forward(state)


class Critic(nn.Module):

    def __init__(self, input_size, hidden_size=128):
        super(Critic, self).__init__()
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )
        self.apply(weights_init_)

    def forward(self, x):
        return self.net(x)


class TwinCritic(nn.Module):

    def __init__(self, input_size, hidden_size=128):
        super(TwinCritic, self).__init__()
        self.net1 = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )

        self.net2 = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )

    def forward(self, x):
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2
