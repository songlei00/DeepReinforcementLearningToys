import gym
import torch
from torch import nn, optim, Tensor
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from collections import namedtuple
import math
import random

def normal_logprob(x, mean, logstd): # 代入正态分布公式计算概率
    std = torch.exp(logstd)
    std_sq = std.pow(2)
    logprob = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
    return logprob.sum(1)

class ActorCriticNet(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(ActorCriticNet, self).__init__()

        self.a_fc1 = nn.Linear(n_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)
        self.a_fc3 = nn.Linear(64, n_outputs)
        self.a_logstd = nn.Parameter(torch.zeros(1, n_outputs))

        self.c_fc1 = nn.Linear(n_inputs, 64)
        self.c_fc2 = nn.Linear(64, 64)
        self.c_fc3 = nn.Linear(64, 1)

        self._norm_layer(self.a_fc1)
        self._norm_layer(self.a_fc2)
        self._norm_layer(self.a_fc3, std=0.01)
        
        self._norm_layer(self.c_fc1)
        self._norm_layer(self.c_fc2)
        self._norm_layer(self.c_fc3)

    def forward(self, state):
        a_mean, a_logstd = self._forward_actor(state)
        c_val = self._forward_critic(state)
        return a_mean, a_logstd, c_val

    def _forward_actor(self, state):
        x = torch.tanh(self.a_fc1(state))
        x = torch.tanh(self.a_fc2(x))
        a_mean = torch.tanh(self.a_fc3(x))
        a_logstd = self.a_logstd.expand_as(a_mean)
        return a_mean, a_logstd

    def _forward_critic(self, state):
        x = torch.tanh(self.c_fc1(state))
        x = torch.tanh(self.c_fc2(x))
        c_val = self.c_fc3(x)
        return c_val

    def _norm_layer(self, layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def get_logprob(self, state, action):
        a_mean, a_logstd = self._forward_actor(state)
        logprob = normal_logprob(action, a_mean, a_logstd)
        return logprob


class DataBuffer:

    def __init__(self, name, *args):
        self.data = namedtuple(name, args)
        self.buf = []

    def push(self, *args):
        self.buf.append(self.data(*args))

    def __len__(self):
        return len(self.buf)


class PPO:

    def __init__(
        self,
        env,
        max_sample_epoch=20,
        max_sample_step=5000,
        batch_sz=256,
        n_update=10,
        save_per_epoch=100,
        clip=0.2,
        learning_rate=0.002,
        gamma=0.995,
        lambda_=0.97,
        seed=1
    ):
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.max_sample_epoch = max_sample_epoch
        self.max_sample_step = max_sample_step
        self.batch_sz = batch_sz
        self.n_update = n_update
        self.save_per_epoch = save_per_epoch
        self.clip = clip
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.seed = seed

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.net = ActorCriticNet(self.n_state, self.n_action).to(self.device)
        # self.net.load_state_dict(torch.load('model/ppo_10.pth'))
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.reward_list = []

    def select_action(self, a_mean, a_logstd):
        a_std = torch.exp(a_logstd)
        a = torch.normal(a_mean, a_std)
        return a[0].cpu().detach().numpy(), normal_logprob(a, a_mean, a_logstd).cpu().detach().numpy()

    def running_state(self, state):
        state -= np.mean(state)
        state /= (np.std(state) + 1e-8)
        return state

    def get_KL_divergence(self, logprob_1, logprob_2):
        return -(torch.exp(logprob_1) * (logprob_1 - logprob_2))

    def evaluate(self, n = 1):
        tot_reward = 0
        for _ in range(n):
            s = self.env.reset()
            while True:
                # self.env.render()
                s = self.running_state(s)
                a_mean, a_logstd, val = self.net(torch.Tensor(s).unsqueeze(0).to(self.device))
                a, logprob = self.select_action(a_mean, a_logstd)
                s_, r, done, _ = self.env.step(a)
                tot_reward += r

                if done:
                    break
                s = s_

        return tot_reward / n

    def train(self, epochs):
        for epoch in range(epochs):
            trace = DataBuffer(
                'trace_data',
                'state',
                'action',
                'reward',
                'next_state',
                'done',
                'action_mean',
                'action_logstd',
                'action_logprob',
                'value',
                'tot_reward',
                'advantage'
            )

            # 采样数据
            for sample_epoch in range(self.max_sample_epoch):
                tmp_trace = DataBuffer(
                    'tmp_trace',
                    'state',
                    'action',
                    'reward',
                    'next_state',
                    'done',
                    'action_mean',
                    'action_logstd',
                    'action_logprob',
                    'value'
                )
                s = self.env.reset()

                # 采集一轮数据
                for _ in range(self.max_sample_step):
                    # self.env.render()
                    s = self.running_state(s)
                    a_mean, a_logstd, val = self.net(torch.Tensor(s).unsqueeze(0).to(self.device))
                    print(a_logstd.exp())
                    a, logprob = self.select_action(a_mean, a_logstd)
                    s_, r, done, _ = self.env.step(a)

                    tmp_trace.push(s, a, r, s_, done, a_mean.cpu().detach().numpy(), a_logstd.cpu().detach().numpy(), logprob, val.cpu().detach().numpy()[0][0])

                    if done:
                        break
                    s = s_

                # 计算total reward和advantage
                trace_len = len(tmp_trace)
                tot_reward_list = np.zeros(trace_len)
                advantage_list = np.zeros(trace_len)
                deltas = np.zeros(trace_len)
                tot_reward = 0.0
                prev_value = 0.0
                prev_advantage = 0.0
                for i in reversed(range(trace_len)):
                    tot_reward = tmp_trace.buf[i].reward + self.gamma * tot_reward
                    tot_reward_list[i] = tot_reward
                    deltas[i] = tmp_trace.buf[i].reward + self.gamma * prev_value - tmp_trace.buf[i].value
                    advantage_list[i] = deltas[i] + self.gamma * self.lambda_ * prev_advantage 
                    # advantage_list[i] = (advantage_list[i] - np.mean(advantage_list[i])) / (np.std(advantage_list[i]) + 1e-8)

                    prev_value = tmp_trace.buf[i].value
                    prev_advantage = advantage_list[i]

                # 记录数据
                for i in range(trace_len):
                    trace.push(*tuple(tmp_trace.buf[i]), tot_reward_list[i], advantage_list[i])

            # print('finish sample')
            # 更新网络
            for _ in range(self.n_update):
                # print(len(trace))
                batch_trace = random.sample(trace.buf, k=min(self.batch_sz, len(trace)))
                batch_state = Tensor([i.state for i in batch_trace]).to(self.device)
                batch_action = Tensor([i.action for i in batch_trace]).to(self.device)
                batch_value = Tensor([i.value for i in batch_trace]).to(self.device)
                batch_oldlogprob = Tensor([i.action_logprob for i in batch_trace]).to(self.device)
                batch_newlogprob = self.net.get_logprob(batch_state, batch_action)
                batch_advantage = Tensor([i.advantage for i in batch_trace]).to(self.device)
                batch_tot_reward = Tensor([i.tot_reward for i in batch_trace]).to(self.device)
                batch_newvalue = self.net._forward_critic(batch_state)

                policy_ratio = torch.exp(batch_newlogprob - batch_oldlogprob)
                surrogate_1 = policy_ratio * batch_advantage
                surrogate_2 = policy_ratio.clamp(1-self.clip, 1+self.clip) * batch_advantage
                loss_surr = torch.mean(torch.min(surrogate_1, surrogate_2))

                loss_critic = torch.mean((batch_newvalue - batch_tot_reward).pow(2))
                loss_kl = torch.mean(self.get_KL_divergence(batch_newlogprob, batch_oldlogprob))
                loss_entropy = torch.mean(torch.exp(batch_newlogprob) * batch_newlogprob)

                loss = - loss_surr + 0.5 * loss_critic 
                # loss = - loss_surr + 0.5 * loss_critic + 0.01 * loss_entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # print('finish update')

            # ep_ratio = 1 - (epoch / epochs)
            # self.clip = 0.2 * ep_ratio

            # 评估网络
            eval_reward = self.evaluate(5)
            self.reward_list.append(eval_reward)
            print('epoch', epoch, ':', eval_reward)

            if epoch % self.save_per_epoch == self.save_per_epoch-1:
                PATH = 'model/ppo_%d.pth'%(epoch)
                torch.save(self.net.state_dict(), PATH)

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('PPO')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('PPO.png')
        plt.show()

epochs = 100
# env = gym.make('HalfCheetah-v2')
env = gym.make('Hopper-v2')

ppo = PPO(
    env
)
ppo.train(700)
ppo.plot_reward()

# for epoch in range(epochs):
#     s = env.reset()
#     tot_r = 0

#     while True:
#         env.render()
#         a = env.action_space.sample()
#         print(a)
#         s_, r, done, _ = env.step(a)
#         # print(len(s_), r, done)
#         tot_r += r

#         s = s_
#         if done:
#             break
#     # print(tot_r)