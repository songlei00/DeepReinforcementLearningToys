import gym
from collections import namedtuple, deque
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

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

class QNetwork(nn.Module):

    def __init__(self, input_sz=4, output_sz=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_sz, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, output_sz)

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
        epsilon=0.1,
        target_update=10,
        max_step=10000
    ):
        self.env = env
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.max_step = max_step

        self.replay_buffer = ReplayBuffer(200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork()    # q eval
        self.q_target = QNetwork()  # fixed
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.RMSprop(self.q_net.parameters())
        self.loss_fn = nn.MSELoss()

        self.reward_list = []
        

    def select_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            with torch.no_grad():
                a = self.q_net(state).max(1)[1].numpy()[0]
        return a

    def train(self, spisodes):
        for episode in range(spisodes):
            dqn.step(episode)

    def step(self, episode):
        s = self.env.reset()
        tot_r = 0
        for i in range(self.max_step):
            self.env.render()
            a = self.select_action(s)
            s_, r, done, _ = self.env.step(a)
            # x, x_dot, theta, theta_dot = s_
            # r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            # r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
            # r = r1 + r2
            tot_r += r
            self.replay_buffer.add_experience(s, a, r, s_)
            s = s_
            self.learn()
            if done:
                break

            if episode % self.target_update == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())

        self.reward_list.append(tot_r)

    def learn(self):
        ret = self.replay_buffer.random_select()
        s, a, r, s_ = ret[0]
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        q_val = self.q_net(s)
        s_ = torch.tensor(s_, dtype=torch.float32).unsqueeze(0)
        q_target_val = self.q_target(s_)
        epected_q = q_target_val * self.gamma + r
        loss = self.loss_fn(q_val, epected_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.show()


env = gym.make('CartPole-v0')
dqn = DQN(env=env, n_action=2)

dqn.train(1000)

env.close()

dqn.plot_reward()
