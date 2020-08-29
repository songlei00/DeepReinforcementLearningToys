from tools.memory import Memory 
from tools.agent import BaseAgent
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import gym
import matplotlib.pyplot as plt
import numpy as np 
import shutil


def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)


class DeterministicPolicy(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(DeterministicPolicy, self).__init__()
        # TODO: add BN layer
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
            nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, x):
        return self.net(x)


class QFunction(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(QFunction, self).__init__()
        # TODO: add BN layer
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )
        self.apply(weight_init)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)


class OUNoise:
    """Ornstein-Uhlenbeck Process"""

    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu-self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class DDPG(BaseAgent):

    def __init__(
        self,
        env_name,
        env,
        batch_size=64,
        gamma=0.99,
        q_lr=1e-3,
        policy_lr=1e-4,
        start_step=1000,
        target_update_interval=10,
        evaluate_interval=10,
        tau=1e-3,
        theta=0.15,
        sigma=0.2,
        replay_buffer_size=1e6
    ): 
        BaseAgent.__init__(self)
        self.env_name = env_name
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.start_step = start_step
        self.target_update_interval = target_update_interval
        self.evaluate_interval = evaluate_interval
        self.tau = tau
        self.theta = theta
        self.sigma = sigma
        self.replay_buffer_size = replay_buffer_size

        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        
        self.memory = Memory(
            'memory',
            int(self.replay_buffer_size),
            self.batch_size,
            'state',
            'action',
            'reward',
            'next_state',
            'mask'
        )

        self.actor = DeterministicPolicy(self.n_state, self.n_action).to(self.device)
        self.target_actor = DeterministicPolicy(self.n_state, self.n_action).to(self.device)
        self.update_target(self.target_actor, self.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.q_lr)

        self.critic = QFunction(self.n_state + self.n_action, 1).to(self.device)
        self.target_critic = QFunction(self.n_state + self.n_action, 1).to(self.device)
        self.update_target(self.target_critic, self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.policy_lr)

        self.mse_loss = nn.MSELoss()
        self.ou_noise = OUNoise(np.zeros(self.n_action), sigma=self.sigma, theta=self.theta)

    def select_action(self, state, is_evaluate=False):
        state = Tensor(state).unsqueeze(0).to(self.device)
        if is_evaluate:
            a = self.actor(state)
        else:
            a = self.actor(state) + Tensor(self.ou_noise()).to(self.device)
        return a.detach().cpu().numpy()[0]

    def update_actor(self, batch):
        """J = (r + gamma * Q'(s_{t+1}, mu'(s_{t+1}) - Q(s_t, a_t)))^2/N"""
        batch_state, batch_action, batch_reward, batch_next_state, batch_mask = batch
        batch_next_action = self.target_actor(batch_next_state).detach()
        target_q = batch_reward + self.gamma * batch_mask * self.target_critic(batch_next_state, batch_next_action)
        q = self.critic(batch_state, batch_action) 
        q_loss = self.mse_loss(q, target_q.detach())
        # self.writer.add_scalar('loss/q_loss', q_loss, self.global_step)
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

    def update_critic(self, batch):
        """J = Q(s_t, mu(s_t))"""
        batch_state, batch_action, batch_reward, batch_next_state, batch_mask = batch
        policy_loss = -torch.mean(self.critic(batch_state, self.actor(batch_state)))
        # self.writer.add_scalar('loss/policy_loss', policy_loss, self.global_step)
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

    def train(self, epochs):
        for _ in range(epochs):
            s = self.env.reset()

            while True:
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                mask = 0 if done else 1
                self.memory.push(s, a, r, s_, mask)
                self.global_step += 1

                if self.global_step < self.start_step or len(self.memory) < self.batch_size:
                    continue

                batch_state, batch_action, batch_reward, batch_next_state, batch_mask = self.memory.sample()
                batch_state = Tensor(batch_state).to(self.device)
                batch_action = Tensor(batch_action).to(self.device)
                batch_reward = Tensor(batch_reward).unsqueeze(1).to(self.device)
                batch_next_state = Tensor(batch_next_state).to(self.device)
                batch_mask = Tensor(batch_mask).unsqueeze(1).to(self.device)
                
                self.update_critic((batch_state, batch_action, batch_reward, batch_next_state, batch_mask))
                self.update_actor((batch_state, batch_action, batch_reward, batch_next_state, batch_mask))

                self.update_target(self.target_critic, self.critic, self.tau)
                self.update_target(self.target_actor, self.actor, self.tau)

                if done:
                    break
                s = s_

            self.global_epoch += 1
            # print('Finish epoch:', self.global_epoch)

            if self.global_epoch % self.evaluate_interval == 0:
                eval_r = self.evaluate()
                self.writer.add_scalar('total_reward', eval_r, self.global_epoch)
                print('epoch', self.global_epoch, eval_r)


if __name__ == '__main__':
    # env = gym.make('Pendulum-v0')
    env = gym.make('HalfCheetah-v2')
    agent = DDPG(
        'DDPG',
        env
    )
    agent.train(2000)
    agent.plot_reward()