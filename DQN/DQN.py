import gym
from collections import namedtuple, deque
from torch import nn, optim
import random

config = {
    'name': None
}

class ReplayBuffer():

    def __init__(self, buf_sz, batch_sz=1, seed=None):
        self.memory = deque(maxlen=buf_sz)
        self.batch_sz = batch_sz
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        random.seed(seed)

    def add_experience(self, s, a, r, s_, done):
        experience = self.experience(s, a, r, s_, done)
        self.memory.append(experience)

    def random_select(self, n=None):
        k = n if n != None else self.batch_sz
        return random.sample(self.memory, k=k)

class BaseAgent():

    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
class DQN(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(config)

env = gym.make('CartPole-v0')

for episode in range(10):
    env.reset()
    while True:
        env.render()
        # a = np.random.randint(0, 2)
        a = int(input())
        obs, r, done, _ = env.step(a)
        print(obs, r, done)
        if done:
            break

env.close()
