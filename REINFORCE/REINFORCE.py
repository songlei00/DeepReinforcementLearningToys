import gym
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    
    def __init__(self, input_sz, output_sz):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_sz, 128)
        self.drop1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, output_sz)

    def forward(self, x):
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)
        return x

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

class REINFORCE:

    def __init__(
        self,
        env,
        n_state,
        n_action,
        gamma=0.9,
        learn_rate=0.015,
        epsilon=0.1,
        is_save_fig=True,
        is_with_baseline=True,
    ):
        self.env = env
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.is_save_fig = is_save_fig
        self.is_with_baseline = is_with_baseline

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNet(self.n_state, self.n_action).to(self.device)
        
        if self.is_with_baseline:
            self.baseline_net = SimpleNet(self.n_state, self.n_action).to(self.device)
            self.opt = optim.Adam([{'params': self.policy_net.parameters()}, {'params': self.baseline_net.parameters()}], lr=self.learn_rate)
        else:
            self.opt = optim.Adam(self.policy_net.parameters(), lr=self.learn_rate)

        self.reward_list = []

    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32, device=self.device), 0)
        a = np.random.choice(range(self.n_action), p=self.policy_net(state).detach()[0].cpu().numpy())
        return a

    def train(self, epochs):
        for epoch in range(epochs):
            s = self.env.reset()
            tot_r = 0
            trajectory = []

            while True:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                tot_r += r

                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                trajectory.append((s, a, r))
                s = s_

                if done:
                    break

            self.learn(trajectory)
            self.reward_list.append(tot_r)

    def learn(self, trajectory):
        G = 0
        pg_loss = 0
        if self.is_with_baseline:
            bl_loss = 0

        for t in reversed(range((len(trajectory)))):
            s, a, r = trajectory[t]
            G += (self.gamma ** t) * r
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32, device=self.device), 0)
            if self.is_with_baseline:
                bl_val = torch.mm(self.baseline_net(s)[0].view(1, -1), self.policy_net(s)[0].view(-1, 1))[0][0]
                td_err = (G - bl_val).detach()
                pg_loss -= (self.gamma ** t) * td_err * torch.log(self.policy_net(s)[0][a])
                bl_loss -= td_err * bl_val
            else:
                pg_loss -= G * torch.log(self.policy_net(s)[0][a]) # 似乎少个系数

        if self.is_with_baseline:
            loss = pg_loss + bl_loss
        else:
            loss = pg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('REINFORCE: CartPole-v1')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        if self.is_save_fig:
            plt.savefig('REINFORCE_CartPole.png')
        plt.show()

env = gym.make('CartPole-v1')

reinforce_with_bl = REINFORCE(
    env=env,
    n_state=env.observation_space.shape[0],
    n_action=env.action_space.n,
    is_save_fig=False,
    is_with_baseline=True
)

reinforce_with_bl.train(500)

env.close()

reinforce_with_bl.plot_reward()
