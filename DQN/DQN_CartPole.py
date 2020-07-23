import sys
sys.path.append('..')

from tools.replay_buffer import ReplayBuffer
import gym
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

class SimpleNet(nn.Module):

    def __init__(self, input_sz, output_sz):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_sz, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, output_sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
class DQN():

    def __init__(
        self,
        env,
        n_state,
        n_action,
        gamma=0.9,
        buf_sz=2000,
        batch_sz=32,
        learn_rate=0.01,
        epsilon=0.1,
        target_update=100,
    ):
        self.env = env
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.buf_sz = buf_sz
        self.batch_sz = batch_sz
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer(buf_sz, batch_sz=batch_sz)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = SimpleNet(self.n_state, self.n_action).to(self.device)     # q eval
        self.q_target = SimpleNet(self.n_state, self.n_action).to(self.device)  # fixed q
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=learn_rate)
        self.loss_fn = nn.MSELoss()

        self.step_cnt = 0
        self.reward_list = []

    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32, device=self.device), 0)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_action)
        else:
            a = self.q_net(state).max(1)[1].cpu().numpy()[0]
        return a

    def train(self, epochs):
        for epoch in range(epochs):
            s = self.env.reset()
            tot_r = 0

            while True:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                tot_r += r

                # state为(车的位置，车的速度，杆的角度，杆的速度)
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
                
                self.replay_buffer.add_experience(s, a, r, s_)
                s = s_

                self.step_cnt += 1
                if self.step_cnt >= self.buf_sz:
                    self.learn()

                if done:
                    break
                
            self.reward_list.append(tot_r)

    def learn(self):
        if self.step_cnt % self.target_update == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
            # print('Target net update')

        # 获得历史数据
        batch_data = self.replay_buffer.random_select()
        s_batch = torch.tensor([i.state for i in batch_data], dtype=torch.float32, device=self.device)
        a_batch = torch.tensor([i.action for i in batch_data], dtype=torch.long, device=self.device)
        r_batch = torch.tensor([i.reward for i in batch_data], dtype=torch.float32, device=self.device)
        next_s_batch = torch.tensor([i.next_state for i in batch_data], dtype=torch.float32, device=self.device)

        # 更新
        q_val = self.q_net(s_batch).gather(dim=1, index=a_batch.view(-1, 1)).view(self.batch_sz)
        q_target_val = self.q_target(next_s_batch).detach().max(dim=1)[0] * self.gamma + r_batch

        loss = self.loss_fn(q_val, q_target_val)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('DQN: CartPole-v1')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('DQN_CartPole.png')
        plt.show()


env = gym.make('CartPole-v1')
dqn = DQN(
    env=env,
    n_state=env.observation_space.shape[0],
    n_action=env.action_space.n
)

dqn.train(300)

env.close()

dqn.plot_reward()
