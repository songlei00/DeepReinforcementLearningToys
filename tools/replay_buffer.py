from collections import namedtuple, deque
import random

class ReplayBuffer():

    def __init__(self, buf_sz, batch_sz=1, seed=None):
        self.memory = deque(maxlen=buf_sz)
        self.batch_sz = batch_sz
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state'])
        random.seed(seed)

    def add_experience(self, s, a, r, s_):
        experience = self.experience(s, a, r, s_)
        self.memory.append(experience)

    def random_select(self, n=None):
        k = n if n != None else self.batch_sz
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)