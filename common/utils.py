from collections import namedtuple, deque
import random
import os
import torch
import numpy as np


class Memory:
    def __init__(self, maxlen, sample_size):
        self.experience = namedtuple(
            'experience',
            (
                'state',
                'action',
                'reward',
                'next_state',
                'done',
            )
        )
        self.memory = deque(maxlen=maxlen)
        self.sample_size = sample_size

    def push(self, *args):
        self.memory.append(self.experience(*args))

    def sample(self):
        return self.experience(*zip(*random.sample(self.memory, self.sample_size)))

    def __len__(self):
        return len(self.memory)


class OUNoise:

    def __init__(self,
                 size,
                 mu=0.0,
                 theta=0.15,
                 sigma=0.2,
                 dt=1e-2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(size)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state += dx
        return self.state

    def __call__(self):
        return self.sample()

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# http://www.johndcook.com/blog/standard_deviation/
class RunningStat:

    def __init__(self, shape):
        self.n = 0
        self.m = np.zeros(shape)
        self.s = np.zeros(shape)

    def push(self, x):
        x = np.array(x)
        assert x.shape == self.m.shape
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m.copy()
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    @property
    def mean(self):
        return self.m

    @property
    def var(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros(self.m.shape)

    @property
    def std(self):
        return np.sqrt(self.var)


class ZFilter:

    def __init__(self, shape, clip=None):
        self.rs = RunningStat(shape)
        self.clip = clip

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        x = (x - self.rs.mean) / (self.rs.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x


def save_model(model, path):
    p = os.path.dirname(path)
    if not os.path.exists(p):
        os.makedirs(p)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def update_model(target_model, model, tau=1):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
