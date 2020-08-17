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

class AC:

    def __init__(
        self,
        env,
        n_state,
        n_action,
        gamma=0.99,
        learn_rate=0.001,
        epsilon=0.1,
        is_save_fig=True,
    ):
        self.env = env
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.is_save_fig = is_save_fig

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_net = PolicyNet(self.n_state, self.n_action).to(self.device)
        self.critic_net = SimpleNet(self.n_state, self.n_action).to(self.device)
        self.opt = optim.Adam([{'params': self.actor_net.parameters()}, {'params': self.critic_net.parameters()}], lr=self.learn_rate)

        self.reward_list = []

    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32, device=self.device), 0)
        a = np.random.choice(range(self.n_action), p=self.actor_net(state).detach()[0].cpu().numpy())
        return a

    def train(self, epochs):
        for epoch in range(epochs):
            s = self.env.reset()
            tot_r = 0
            trajectory = []

            I = 1
            while True:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                tot_r += r

                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                self.learn(s, a, r, s_, I)
                I = self.gamma * I
                s = s_

                if done:
                    break

            self.reward_list.append(tot_r)

    def learn(self, s, a, r, s_, I):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32, device=self.device), 0)
        s_val = torch.mm(self.critic_net(s)[0].view(1, -1), self.actor_net(s)[0].view(-1, 1))[0][0]
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float32, device=self.device), 0)
        next_s_val = torch.mm(self.critic_net(s_)[0].view(1, -1), self.actor_net(s_)[0].view(-1, 1))[0][0]
        td_err = (r + next_s_val - s_val).detach()

        a_loss = I * td_err * torch.log(self.actor_net(s)[0][a])
        c_loss = td_err * s_val

        loss = a_loss + c_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('Actor Critic: CartPole-v1')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        if self.is_save_fig:
            plt.savefig('Actor_Critic_CartPole.png')
        plt.show()

env = gym.make('CartPole-v1')

ac = AC(
    env=env,
    n_state=env.observation_space.shape[0],
    n_action=env.action_space.n,
    is_save_fig=False,
)

ac.train(500)

env.close()

ac.plot_reward()
