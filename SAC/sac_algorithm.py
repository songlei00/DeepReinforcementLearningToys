import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import namedtuple

import sys
sys.path.append('..')
from common.network import SACGaussianActor, TwinCritic
from common.utils import Memory, ZFilter, update_model, save_model


class ExtendMemory(Memory):

    def __init__(self, maxlen, sample_size):
        Memory.__init__(self, maxlen, sample_size)
        self.experience = namedtuple(
            'experience',
            (
                'state',
                'action',
                'log_prob',
                'reward',
                'next_state',
                'done',
            )
        )


class SAC:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-3,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 batch_size=32,
                 replay_memory_size=1e6,
                 update_frequency=2,
                 warmup_step=1e3,
                 tau=0.005,
                 alpha=None,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=1,
                 save_log_frequency=20):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.update_frequency = update_frequency
        self.warmup_step = warmup_step
        self.tau = tau
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency
        self.save_log_frequency = save_log_frequency

        self.total_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/SAC_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss
        self.memory = ExtendMemory(int(replay_memory_size), batch_size)

        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0]
        self.state_normalize = ZFilter(n_state)
        if alpha is None:
            self.auto_tune_alpha = True
            self.target_entropy = - torch.prod(torch.Tensor(env.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            print('Auto adjust alpha')
        else:
            self.auto_tune_alpha = False
            self.log_alpha = torch.log(torch.tensor(alpha, dtype=torch.float)).to(self.device)
            print('Fixed alpha')

        self.actor = SACGaussianActor(n_state, n_action, 128, action_scale=int(env.action_space.high[0])).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = TwinCritic(n_state + n_action).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.target_critic = TwinCritic(n_state + n_action).to(self.device)
        update_model(self.target_critic, self.critic)

        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # BUG: 最好是对state进行unsqueeze后再输出，保证和batch的数据的输入维度相同
        a, log_prob = self.actor.sample(state, is_test)
        return a.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            s = self.env.reset()
            s = self.state_normalize(s)
            policy_loss, critic_loss, alpha_loss = 0, 0, 0
            while True:
                self.env.render()
                a, log_prob = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                self.memory.push(s, a, log_prob, r, s_, done)
                self.total_step += 1
                if len(self.memory) > self.batch_size and self.total_step > self.warmup_step:
                    policy_loss, critic_loss, alpha_loss = self.learn()

                s = s_
                if done:
                    break

            if (epoch + 1) % self.save_log_frequency == 0:
                self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_step)
                self.writer.add_scalar('loss/policy_loss', policy_loss, self.total_step)
                self.writer.add_scalar('alpha', self.log_alpha.exp(), self.total_step)
                self.writer.add_scalar('loss/alpha_loss', alpha_loss, self.total_step)

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
        batch = self.memory.sample()
        batch_state, batch_action, batch_log_prob, batch_reward, batch_next_state, batch_done = \
            batch.state, batch.action, batch.log_prob, batch.reward, batch.next_state, batch.done
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.float).reshape(self.batch_size, -1).to(self.device)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).reshape(self.batch_size, -1).to(self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(self.device)
        batch_mask = torch.tensor([not i for i in batch_done], dtype=torch.bool).reshape(self.batch_size, -1).to(
            self.device)

        alpha = self.log_alpha.exp()
        # update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch_next_state)
            next_log_prob = next_log_prob.sum(1, keepdim=True)
            next_input = torch.cat([batch_next_state, next_action], dim=-1)
            target_q1, target_q2 = self.target_critic(next_input)
            target_q = batch_reward + batch_mask * self.gamma * (torch.min(target_q1, target_q2) - alpha * next_log_prob)

        q1, q2 = self.critic(torch.cat([batch_state, batch_action], dim=-1))
        critic_loss_1 = self.loss_fn(q1, target_q)
        critic_loss_2 = self.loss_fn(q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        # batch_pi, batch_pi_log_prob = self.actor.sample(batch_state)
        # q1, q2 = self.critic(torch.cat([batch_state, batch_pi], dim=-1))
        # batch_pi_log_prob = batch_pi_log_prob.sum(1, keepdim=True)
        # policy_loss = (alpha * batch_pi_log_prob - torch.min(q1, q2)).mean()
        # self.actor_opt.zero_grad()
        # policy_loss.backward()
        # self.actor_opt.step()

        q1, q2 = self.critic(torch.cat([batch_state, batch_action], dim=-1))
        batch_log_prob = batch_log_prob.sum(1, keepdim=True)
        policy_loss = (alpha * batch_log_prob - torch.min(q1, q2)).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        # update alpha
        if self.auto_tune_alpha:
            alpha_loss = - (self.log_alpha * (batch_log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0)

        if (self.total_step+1) % self.update_frequency == 0:
            update_model(self.target_critic, self.critic)

        return policy_loss.item(), critic_loss.item(), alpha_loss.item()

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
