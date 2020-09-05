import sys
sys.path.append('../tools')
from agent import BaseAgent
from memory import Memory
from network import DeterministicPolicy, ClippedDoubleQFunction

import torch
from torch import nn, optim, Tensor
import gym
import numpy as np


class TD3(BaseAgent):

    def __init__(
        self,
        env_name,
        env,
        lr=3e-4,
        update_interval=2,
        evaluate_interval=100,
        gamma=0.99,
        tau=0.005,
        buffer_size=1e6,
        batch_size=256,
        noise_std=np.sqrt(0.2),
        noise_clip=0.5
    ):
        BaseAgent.__init__(self)
        self.env_name = env_name
        self.env = env
        self.lr = lr
        self.update_interval = update_interval
        self.evaluate_interval = evaluate_interval
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.action_scale = env.action_space.high[0]

        self.actor = DeterministicPolicy(self.n_state, self.n_action).to(self.device)
        self.target_actor = DeterministicPolicy(self.n_state, self.n_action).to(self.device)
        self.update_target(self.target_actor, self.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = ClippedDoubleQFunction(self.n_state+self.n_action, 1).to(self.device)
        self.target_critic = ClippedDoubleQFunction(self.n_state+self.n_action, 1).to(self.device)
        self.update_target(self.target_critic, self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.mse_loss = nn.MSELoss()
        self.memory = Memory(
            'memory',
            int(self.buffer_size),
            self.batch_size,
            'state',
            'action',
            'reward',
            'next_state',
            'mask'
        )

    def select_action(self, state, is_evaluate=False):
        state = Tensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        if is_evaluate == False:
            noise = (torch.randn_like(action)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            action += noise
        action.clamp_(-self.action_scale, self.action_scale)

        return action.detach().cpu().numpy()[0]

    def update_critic(self, batch):
        batch_state, batch_action, batch_reward, batch_next_state, batch_mask = batch
        with torch.no_grad():
            noise = (torch.randn_like(batch_action)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            batch_next_action = self.target_actor(batch_next_state) + noise
            target_Q1, target_Q2 = self.target_critic(batch_next_state, batch_next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batch_reward + self.gamma * batch_mask * target_Q

        current_Q1, current_Q2 = self.critic(batch_state, batch_action)
        Q_loss = self.mse_loss(current_Q1, target_Q) + self.mse_loss(current_Q2, target_Q)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        self.critic_opt.step()

    def update_actor(self, batch):
        batch_state, batch_action, batch_reward, batch_next_state, batch_mask = batch
        policy_loss = - self.critic.Q1(batch_state, self.actor(batch_state)).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

    def do_update(self, batch):
        self.update_critic(batch)
        if self.global_epoch % self.update_interval == 0:
            self.update_actor(batch)
            self.update_target(self.target_actor, self.actor, self.tau)
            self.update_target(self.target_critic, self.critic, self.tau)

    def train(self, epochs):
        for _ in range(epochs):
            s = self.env.reset()

            while True:
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                mask = 0 if done else 1
                self.memory.push(s, a, r, s_, mask)
                self.global_step += 1

                if done:
                    break
                s = s_

                if len(self.memory) < self.batch_size:
                    continue

                batch = self.memory.sample()
                batch_state = Tensor(batch.state).to(self.device)
                batch_action = Tensor(batch.action).to(self.device)
                batch_reward = Tensor(batch.reward).unsqueeze(1).to(self.device)
                batch_next_state = Tensor(batch.next_state).to(self.device)
                batch_mask = Tensor(batch.mask).unsqueeze(1).to(self.device)

                self.do_update((batch_state, batch_action, batch_reward, batch_next_state, batch_mask))

            if self.global_epoch % self.evaluate_interval == 0:
                eval_r = self.evaluate()
                self.writer.add_scalar('Reward', eval_r, self.global_epoch)
                print('Finish epoch', self.global_epoch, eval_r)

            self.global_epoch += 1
            

if __name__ == '__main__':
    env = gym.make('Hopper-v2')    
    agent = TD3(
        'TD3',
        env
    )
    agent.train(3000)
