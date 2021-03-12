import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')
from common.network import GaussianActor, Critic
from common.utils import Trace, ZFilter, save_model


def orthogonal_weights_init_(m, std=1.0, bias=1e-6):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, std)
        nn.init.constant_(m.bias, bias)


class PPO:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 sample_size=2048,
                 batch_size=64,
                 sample_reuse=1,
                 train_iters=5,
                 clip=0.2,
                 gamma=0.99,
                 lam=0.95,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=5,
                 save_log_frequency=1):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.sample_reuse = sample_reuse
        self.train_iters = train_iters
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency
        self.save_log_frequency = save_log_frequency

        self.total_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/PPO_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss

        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0]
        self.state_normalize = ZFilter(n_state)
        self.actor = GaussianActor(n_state, n_action, 128, action_scale=int(env.action_space.high[0]),
                                   weights_init_=orthogonal_weights_init_).to(self.device)
        self.critic = Critic(n_state, 128, orthogonal_weights_init_).to(self.device)

        # self.optimizer = optim.Adam([
        #     {'params': self.critic.parameters(), 'lr': self.critic_lr},
        #     {'params': self.actor.parameters(), 'lr': self.actor_lr}
        # ])
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.trace = Trace()

        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        a, log_prob = self.actor.sample(state, is_test)
        return a.cpu().detach().numpy()[0], log_prob

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            num_sample = 0
            self.trace.clear()
            s = self.env.reset()
            s = self.state_normalize(s)
            while True:
                # self.env.render()
                a, log_prob = self.select_action(s)
                log_prob = torch.sum(log_prob, dim=1, keepdim=True)
                v = self.critic(torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device))
                s_, r, done, _ = self.env.step(a)
                s_ = self.state_normalize(s_)
                self.trace.push(s, a, log_prob.cpu().detach().numpy()[0], r, s_, not done, v)
                num_sample += 1
                self.total_step += 1
                s = s_
                if done and num_sample >= self.sample_size:
                    break
                if done:
                    s = self.env.reset()
                    s = self.state_normalize(s)

            policy_loss, critic_loss = self.learn()

            if (epoch + 1) % self.save_log_frequency == 0:
                self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_step)
                self.writer.add_scalar('loss/policy_loss', policy_loss, self.total_step)

            if (epoch + 1) % self.save_model_frequency == 0:
                save_model(self.critic, 'model/{}_model/critic_{}'.format(self.env_name, epoch))
                save_model(self.actor, 'model/{}_model/actor_{}'.format(self.env_name, epoch))
                ZFilter.save(self.state_normalize, 'model/{}_model/rs_{}'.format(self.env_name, epoch))

            if (epoch + 1) % self.eval_frequency == 0:
                eval_r = self.evaluate()
                print('epoch', epoch, 'evaluate reward', eval_r)
                self.writer.add_scalar('reward', eval_r, self.total_step)
                if eval_r > best_eval:
                    best_eval = eval_r
                    save_model(self.critic, 'model/{}_model/best_critic'.format(self.env_name))
                    save_model(self.actor, 'model/{}_model/best_actor'.format(self.env_name))
                    ZFilter.save(self.state_normalize, 'model/{}_model/best_rs'.format(self.env_name))

    def learn(self):
        all_data = self.trace.get()
        data_idx_range = np.arange(len(self.trace))
        adv, total_reward = self.trace.cal_advantage(self.gamma, self.lam)
        adv = adv.reshape(len(self.trace), -1).to(self.device)
        total_reward = total_reward.reshape(len(self.trace), -1).to(self.device)

        all_state = torch.tensor(all_data.state, dtype=torch.float).to(self.device)
        all_action = torch.tensor(all_data.action, dtype=torch.float).reshape(len(self.trace), -1).to(self.device)
        all_log_prob = torch.tensor(all_data.log_prob, dtype=torch.float).reshape(len(self.trace), -1).to(self.device)
        all_value = torch.tensor(all_data.value, dtype=torch.float).reshape(len(self.trace), -1).to(self.device)

        policy_loss, critic_loss = 0, 0
        # train_iters = max(int(self.sample_size * self.sample_reuse / self.batch_size), 1)

        for _ in range(self.train_iters):
            batch_idx = np.random.choice(data_idx_range, self.batch_size, replace=False)
            batch_state = all_state[batch_idx]
            batch_action = all_action[batch_idx]
            batch_log_prob = all_log_prob[batch_idx]
            batch_new_log_prob = self.actor.get_log_prob(batch_state, batch_action)
            batch_new_log_prob = batch_new_log_prob.sum(dim=1, keepdim=True)
            batch_old_value = all_value[batch_idx]
            batch_new_value = self.critic(batch_state)
            batch_adv = adv[batch_idx]
            batch_total_reward = total_reward[batch_idx]

            ratio = torch.exp(batch_new_log_prob - batch_log_prob)
            surr1 = ratio * batch_adv.detach()
            surr2 = ratio.clamp(1-self.clip, 1+self.clip) * batch_adv.detach()
            entropy_loss = (torch.exp(batch_new_log_prob) * batch_new_log_prob).mean()
            policy_loss = - torch.min(surr1, surr2).mean() + 0.01*entropy_loss

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            # print(batch_new_value, batch_total_reward)
            clip_v = batch_old_value + torch.clamp(batch_new_value - batch_old_value, -self.clip, self.clip)
            critic_loss = torch.max(
                self.loss_fn(batch_new_value, batch_total_reward),
                self.loss_fn(clip_v, batch_total_reward),
            )
            critic_loss = critic_loss.mean() / (6 * batch_total_reward.std())
            # critic_loss = self.loss_fn(batch_new_value, batch_total_reward)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # critic loss 太大了吧，这能优化吗
            # loss = policy_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            #
            # self.optimizer.zero_grad()
            # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # self.max_grad_norm = 0.5
            # # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            # self.optimizer.step()

        return policy_loss.item(), critic_loss.item()

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            s = self.state_normalize(s, update=False)
            while True:
                if is_render:
                    self.env.render()
                a, _ = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                s_ = self.state_normalize(s_, update=False)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
