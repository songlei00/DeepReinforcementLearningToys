import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append('..')
from common.network import DeterministicActor, TwinCritic
from common.utils import GaussianNoise, Memory, ZFilter, update_model, save_model


class TD3:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 batch_size=32,
                 replay_memory_size=1e6,
                 actor_update_frequency=2,
                 warmup_step=1e3,
                 tau=0.005,
                 mu=0,
                 std=0.2,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=20):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.actor_update_frequency = actor_update_frequency
        self.warmup_step = warmup_step
        self.tau = tau
        self.mu = mu
        self.std = std
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.total_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/TD3_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss
        self.memory = Memory(int(replay_memory_size), batch_size)

        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0]
        self.noise = GaussianNoise(n_action, mu=self.mu, std=self.std, clip=0.5)
        self.state_normalize = ZFilter(n_state)

        self.actor = DeterministicActor(n_state, n_action, hidden_size=256,
                                        action_scale=int(env.action_space.high[0])).to(self.device)
        self.target_actor = DeterministicActor(n_state, n_action, hidden_size=256,
                                               action_scale=int(env.action_space.high[0])).to(self.device)
        update_model(self.target_actor, self.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = TwinCritic(n_state+n_action, hidden_size=256).to(self.device)
        self.target_critic = TwinCritic(n_state+n_action, hidden_size=256).to(self.device)
        update_model(self.target_critic, self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        if is_test:
            a = self.actor(state)
        else:
            a = self.actor(state) + torch.tensor(self.noise(), dtype=torch.float).to(self.device)
            a = a.clip(-self.actor.action_scale, self.actor.action_scale)
        return a.cpu().detach().numpy()

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            s = self.env.reset()
            s = self.state_normalize(s)
            critic_loss, policy_loss = 0, 0
            while True:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                s_ = self.state_normalize(s_)
                self.memory.push(s, a, r, s_, done)
                if len(self.memory) > self.batch_size and len(self.memory) > self.warmup_step:
                    tmp_critic_loss, tmp_policy_loss = self.learn()
                    critic_loss = tmp_critic_loss.item()
                    policy_loss = tmp_policy_loss.item() if tmp_policy_loss is not None else policy_loss

                s = s_
                if done:
                    break

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
        batch = self.memory.sample()
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            batch.state, batch.action, batch.reward, batch.next_state, batch.done
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.float).reshape(self.batch_size, -1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).reshape(self.batch_size, -1).to(self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(self.device)
        batch_mask = torch.tensor([not i for i in batch_done], dtype=torch.bool).reshape(self.batch_size, -1).to(
            self.device)

        critic_loss, policy_loss = None, None
        # update critic
        with torch.no_grad():
            next_action = self.target_actor(batch_next_state)
            target_q1, target_q2 = self.target_critic(torch.cat([batch_next_state, next_action], dim=-1))
            target_q = batch_reward + self.gamma * batch_mask * torch.min(target_q1, target_q2)
        q1, q2 = self.critic(torch.cat([batch_state, batch_action], dim=-1))
        critic_loss_1 = self.loss_fn(q1, target_q)
        critic_loss_2 = self.loss_fn(q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if (self.total_step + 1) % self.actor_update_frequency == 0:
            # update actor
            q1, _ = self.critic(torch.cat([batch_state, self.actor(batch_state)], dim=-1))
            policy_loss = - q1.mean()

            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            # update target actor and target critic
            update_model(self.target_critic, self.critic, self.tau)
            update_model(self.target_actor, self.actor, self.tau)

        self.total_step += 1

        return critic_loss, policy_loss

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            s = self.state_normalize(s, update=False)
            while True:
                if is_render:
                    self.env.render()
                a = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                s_ = self.state_normalize(s_, update=False)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
