import gym
from collections import namedtuple, deque
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import numpy as np

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

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):

    def __init__(self, input_sz=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_sz, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
     
class DQN():

    def __init__(
        self,
        env,
        n_action,
        gamma=0.99,
        epsilon=1.0,
        max_step=None
    ):
        self.env = env
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon

        self.replay_buffer = ReplayBuffer(200)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)     # q eval
        self.q_target = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        self.loss_fn = nn.MSELoss()
        

    def select_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            with torch.no_grad():
                self.q_net(state)


    def step(self):
        s = self.env.reset()
        while True:
            self.env.render()
            a = self.select_action(s)
            s_, r, done, _ = self.env.step(a)
            self.replay_buffer.add_experience(s, a, r, s_, done)
            s = s_
            self.learn()
            if done:
                break

    def learn(self):
        s, a, r, s_, done = self.replay_buffer.random_select()
        with torch.no_grad():
            q_target_val = 


env = gym.make('CartPole-v0')
dqn = DQN(env, 2)

for episode in range(10):
    dqn.step()

env.close()


# env = gym.make('CartPole-v0')

# for episode in range(10):
#     env.reset()
#     while True:
#         env.render()
#         a = np.random.randint(0, 2)
#         # a = int(input())
#         obs, r, done, _ = env.step(a)
#         print(obs, r, done)
#         if done:
#             break

# env.close()
