from collections import namedtuple, deque
import random
import os
import torch


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
