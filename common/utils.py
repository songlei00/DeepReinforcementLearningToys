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


# class RunningStat:
#
#     def __init__(self, shape):
#         self.n = 0
#         self.mean = np.zeros(shape)
#         self.std = np.zeros(shape)
#
#     def push(self, x):
#         x = np.asarray(x)
#         assert x.shape == self.mean.shape
#         self.n += 1
#         if self.n == 1:
#             self.mean = x
#         else:
#             old_mean = self.mean.copy()
#             self.mean += (x - old_mean) / self.n
#             self.std += (x - old_mean) * (x - self.mean)
#
#
# class ZFilter:
#
#     def __init__(self, shape):
#         self.rs = RunningStat(shape)
#
#     def __call__(self, x, update=True):
#         if update:
#             self.rs.push(x)
#         x = (x - self.rs.mean) / (self.rs.std + 1e-8)
#         return x


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
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
