import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append('..')
from common.network import DeterministicActor, Critic
from common.utils import save_model, update_model, Memory, OUNoise


class DDPG:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-3,
                 gamma=0.99,
                 batch_size=32,
                 replay_memory_size=1e6,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=10):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/DDPG_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss
        self.memory = Memory(int(replay_memory_size), batch_size)

        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0]
        self.noise = OUNoise(n_action)

        self.actor = DeterministicActor(n_state, n_action,
                                        action_scale=int(env.action_space.high[0])).to(self.device)
        self.target_actor = DeterministicActor(n_state, n_action,
                                               action_scale=int(env.action_space.high[0])).to(self.device)
        update_model(self.target_actor, self.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(n_state + n_action).to(self.device)
        self.target_critic = Critic(n_state + n_action).to(self.device)
        update_model(self.target_critic, self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        if is_test:
            a = self.actor(state) + torch.tensor(self.noise(), dtype=torch.float).to(self.device)
        else:
            a = self.actor(state)
        return a.cpu().detach().numpy()

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            s = self.env.reset()
            policy_loss, critic_loss = 0, 0
            while True:
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                self.memory.push(s, a, r, s_, done)
                if len(self.memory) > self.batch_size:
                    policy_loss, critic_loss = self.learn()
                s = s_
                if done:
                    break

            self.writer.add_scalar('loss/actor_loss', policy_loss, epoch)
            self.writer.add_scalar('loss/critic_loss', critic_loss, epoch)

            if (epoch + 1) % self.save_model_frequency == 0:
                save_model(self.critic, 'model/{}_model/critic_{}'.format(self.env_name, epoch))
                save_model(self.actor, 'model/{}_model/actor_{}'.format(self.env_name, epoch))

            if (epoch + 1) % self.eval_frequency == 0:
                eval_r = self.evaluate()
                print('epoch', epoch, 'evaluate reward', eval_r)
                self.writer.add_scalar('reward', eval_r, epoch)
                if eval_r > best_eval:
                    best_eval = eval_r
                    save_model(self.critic, 'model/{}_model/best_critic'.format(self.env_name))
                    save_model(self.actor, 'model/{}_model/best_actor'.format(self.env_name))

    def learn(self):
        batch = self.memory.sample()
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            batch.state, batch.action, batch.reward, batch.next_state, batch.done
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.float).reshape(self.batch_size, 1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).reshape(self.batch_size, 1).to(self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(self.device)
        batch_mask = torch.tensor([not i for i in batch_done], dtype=torch.bool).reshape(self.batch_size, 1).to(self.device)

        # update critic
        pred_q = self.critic(torch.cat((batch_state, batch_action), dim=-1))
        next_action = self.target_actor(batch_next_state)
        next_q = self.target_critic(torch.cat((batch_next_state, next_action), dim=-1))
        pred_target_q = batch_reward + batch_mask * self.gamma * next_q
        critic_loss = self.loss_fn(pred_q, pred_target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        policy_loss = - self.critic(torch.cat((batch_state, self.actor(batch_state)), dim=-1)).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        # update target
        update_model(self.target_critic, self.critic, 0.05)
        update_model(self.target_actor, self.actor, 0.05)

        return policy_loss.item(), critic_loss.item()

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            while True:
                if is_render:
                    self.env.render()
                with torch.no_grad():
                    a = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
