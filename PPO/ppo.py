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

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        init.constant_(layer.bias, bias_const)

        # nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)

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


class Memory(object):
    def __init__(self):
        self.experience = namedtuple(
            'experience',
            (
                'state',
                'action',
                'reward',
                'next_state',
                'mask',
                'log_prob',
                'value'
            )
        )
        self.memory = []

    def push(self, *args):
        self.memory.append(self.experience(*args))

    def sample(self):
        return self.experience(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class PPO:

    def __init__(
        self,
        env,
        max_sample_size=2048,
        max_sample_step=2000,
        batch_sz=256,
        n_update=10,
        clip=0.2,
        learning_rate=3e-4,
        gamma=0.995,
        lambda_=0.97,
        seed=1
    ):
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.max_sample_size = max_sample_size
        self.max_sample_step = max_sample_step
        self.batch_sz = batch_sz
        self.n_update = n_update
        self.clip = clip
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.seed = seed

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.net = ActorCriticNet(self.n_state, self.n_action).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.reward_list = []
        self.total_sample_size = 0

    def select_action(self, a_mean, a_logstd):
        a_std = torch.exp(a_logstd)
        a = torch.normal(a_mean, a_std)
        return a, normal_logprob(a, a_mean, a_logstd)

    def norm_state(self, state):
        '''
        state = (state - mean) / std
        this will speed up the training
        '''
        # 这里也可以通过增量更新维护一个目前所有样本的均值和方差
        # 会取得更好的效果
        state -= np.mean(state)
        state /= (np.std(state) + 1e-8)
        return state

    def evaluate(self, n = 1):
        tot_reward = 0
        for _ in range(n):
            s = self.env.reset()
            while True:
                # self.env.render()
                s = self.norm_state(s)
                a_mean, a_logstd, val = self.net(torch.Tensor(s).unsqueeze(0).to(self.device))
                a, logprob = self.select_action(a_mean, a_logstd)
                a = a.cpu().detach().numpy()[0]
                s_, r, done, _ = self.env.step(a)
                tot_reward += r

                if done:
                    break
                s = s_

        return tot_reward / n

    def train(self, epochs):
        for epoch in range(epochs):
            memory = Memory()

            # 采样数据
            sample_size = 0
            while sample_size < self.max_sample_size:
                s = self.env.reset()
                tot_reward = 0
                s = self.norm_state(s)

                # 采集一轮数据
                for step in range(self.max_sample_step):
                    # self.env.render()
                    a_mean, a_logstd, val = self.net(torch.Tensor(s).unsqueeze(0).to(self.device))
                    a, logprob = self.select_action(a_mean, a_logstd)
                    a = a.cpu().detach().numpy()[0]
                    logprob = logprob.cpu().detach().numpy()[0]
                    s_, r, done, _ = self.env.step(a)
                    s_ = self.norm_state(s_)
                    tot_reward += r
                    mask = 0 if done else 1

                    # 这里需要保证push的state和next state都是标准化过的
                    memory.push(s, a, r, s_, mask, logprob, val)

                    if done:
                        break
                    s = s_
                sample_size += step

            # 计算total reward和advantage
            batch = memory.sample()
            batch_size = len(memory)

            states = Tensor(batch.state)
            actions = Tensor(batch.action)
            rewards = Tensor(batch.reward)
            next_states = Tensor(batch.next_state)
            masks = Tensor(batch.mask)
            oldlogprob = Tensor(batch.log_prob)
            values = Tensor(batch.value)

            tot_rewards = Tensor(batch_size)
            deltas = Tensor(batch_size)
            advantages = Tensor(batch_size)

            prev_tot_reward = 0
            prev_value = 0
            prev_advantage = 0

            for i in reversed(range(batch_size)):
                tot_rewards[i] = rewards[i] + self.gamma * prev_tot_reward * masks[i]
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.gamma * self.lambda_ * prev_advantage * masks[i]

                prev_tot_reward = tot_rewards[i]
                prev_value = values[i]
                prev_advantage = advantages[i]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 更新网络
            for _ in range((self.n_update * batch_size) // self.batch_sz):
                batch_idx = np.random.choice(batch_size, self.batch_sz, replace=False)
                batch_state = states[batch_idx]
                batch_action = actions[batch_idx]
                batch_value = values[batch_idx]
                batch_oldlogprob = oldlogprob[batch_idx]
                batch_newlogprob = self.net.get_logprob(batch_state, batch_action)
                batch_advantage = advantages[batch_idx]
                batch_tot_rewards = tot_rewards[batch_idx]
                # 必须要flatten保证batch_newvalue和batch_tot_rewards的size相同
                # 不同的size会在求loss时广播得到错误的loss值，并且有的时候pytorch不会给出warning
                # batch_newvalue = self.net._forward_critic(batch_state)
                batch_newvalue = self.net._forward_critic(batch_state).flatten() 

                policy_ratio = torch.exp(batch_newlogprob - batch_oldlogprob)
                surrogate_1 = policy_ratio * batch_advantage
                surrogate_2 = policy_ratio.clamp(1-self.clip, 1+self.clip) * batch_advantage
                loss_surr = torch.mean(torch.min(surrogate_1, surrogate_2))

                loss_critic = torch.mean((batch_newvalue - batch_tot_rewards).pow(2))
                loss_entropy = torch.mean(torch.exp(batch_newlogprob) * batch_newlogprob)

                loss = - loss_surr + 0.5 * loss_critic - 0.01 * loss_entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # 评估网络
            eval_reward = self.evaluate()
            self.reward_list.append(eval_reward)
            print('epoch', epoch, ':', eval_reward)

    def plot_reward(self):
        plt.plot(self.reward_list)
        plt.title('PPO')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('PPO.png')
        plt.show()


if __name__ == '__main__':

    env = gym.make('HalfCheetah-v2')
    # env = gym.make('Hopper-v2')

    ppo = PPO(
        env
    )
    ppo.train(5000)
    ppo.plot_reward()
