import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F 
import gym
import matplotlib.pyplot as plt
import numpy as np 

from collections import namedtuple, deque
import random


class ReplayBuffer():

    def __init__(self, buf_sz, batch_sz=1):
        self.memory = deque(maxlen=buf_sz)
        self.batch_sz = batch_sz
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state'])

    def push(self, s, a, r, s_):
        experience = self.experience(s, a, r, s_)
        self.memory.append(experience)

    def sample(self, n=None):
        k = n if n != None else self.batch_sz
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # cliped double Q
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None, log_std_min=-20, log_std_max=-2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Reparameterization trick
        eps = torch.randn_like(mean)
        u = eps * std + mean # u ~ N(mean, std^2)
        action = torch.tanh(u).to(self.device)
        # Enforcing action bounds
        log_prob = normal.log_prob(u) - torch.log((1 - action.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class SAC:
    
    def __init__(
        self,
        env,
        batch_sz=256,
        start_step=10000,
        target_update_interval=1,
        n_updates=1,
        gamma=0.99,
        tau=0.005,
        lr=0.0003,
        alpha=0.2
    ):
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.batch_sz = batch_sz
        self.start_step = start_step
        self.target_update_interval = target_update_interval
        self.n_updates = n_updates
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.alpha = alpha

        self.global_step = 0
        self.reward_list = []
        self.memory = ReplayBuffer(1000000, self.batch_sz)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.critic = QNetwork(self.n_state, self.n_action).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(self.n_state, self.n_action).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        self.policy = GaussianPolicy(self.n_state, self.n_action, 256, self.env.action_space).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        
    def select_action(self, state):
        state = Tensor(state).to(self.device).unsqueeze(0)
        action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self):
        batch_data = self.memory.sample()
        batch_state = Tensor([i.state for i in batch_data]).to(self.device)
        batch_action = Tensor([i.action for i in batch_data]).to(self.device)
        batch_reward = Tensor([i.reward for i in batch_data]).to(self.device).unsqueeze(1)
        batch_next_state = Tensor([i.next_state for i in batch_data]).to(self.device)

        # update critic network
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(batch_next_state)
            qf1_next_target, qf2_next_target = self.critic_target(batch_next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = batch_reward + self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(batch_state, batch_action)
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value) 
        qf_loss = qf1_loss + qf2_loss

        self.critic_opt.zero_grad()
        qf_loss.backward()
        self.critic_opt.step()

        # update policy network
        action, log_pi = self.policy.sample(batch_state)
        qf1_pi, qf2_pi = self.critic(batch_state, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # update target network
        if self.global_step % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1-self.tau) + param.data * self.tau)

    def train(self, epochs=100):
        for epoch in range(epochs):
            s = self.env.reset()

            while True:
                # self.env.render()
                if self.global_step < self.start_step:
                    a = self.env.action_space.sample() 
                else:
                    a = self.select_action(s)

                if len(self.memory) > self.batch_sz:
                    for _ in range(self.n_updates):
                        self.update_parameters()

                s_, r, done, _ = self.env.step(a)
                self.memory.push(s, a, r, s_)
                self.global_step += 1
                if done:
                    break
                s = s_
            if epoch % 10 == 0:
                eval_r = self.evaluate()
                print(epoch, ':', eval_r)

    def evaluate(self, n=1):
        tot_reward = 0
        for _ in range(n):
            s = self.env.reset()
            
            while True:
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                tot_reward += r
                if done:
                    break
                s = s_
        
        tot_reward /= n
        self.reward_list.append(tot_reward)
        return tot_reward

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('SAC')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('SAC.png')
        plt.show()

env = gym.make('HalfCheetah-v2')

sac = SAC(env)
sac.train(700)
sac.plot_reward()
env.close()