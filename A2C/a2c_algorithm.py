import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append('..')
from common.network import GaussianActor, Actor, Critic
from common.utils import save_model


class A2C:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-3,
                 gamma=0.99,
                 is_continue_action_space=False,
                 reward_shapeing_func=lambda x: x[1],
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=10):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.reward_shapeing_func = reward_shapeing_func
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/A2C_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss

        if is_continue_action_space:
            self.actor = GaussianActor(env.observation_space.shape[0], env.action_space.shape[0],
                                       action_scale=int(env.action_space.high[0])).to(self.device)
        else:
            self.actor = Actor(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = Critic(env.observation_space.shape[0]).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        a, log_prob = self.actor.sample(state, is_test)
        return a.cpu().detach().numpy(), log_prob

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            s = self.env.reset()
            while True:
                self.env.render()
                a, log_prob = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                r = self.reward_shapeing_func((s_, r, done, _))
                policy_loss, critic_loss = self.learn(s, a, log_prob, r, s_, done)
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

    def learn(self, s, a, log_prob, r, s_, done):
        mask = not done
        next_q = self.critic(torch.tensor(s_, dtype=torch.float).to(self.device))
        target_q = r + mask * self.gamma * next_q
        pred_v = self.critic(torch.tensor(s, dtype=torch.float).to(self.device))
        critic_loss = self.loss_fn(pred_v, target_q.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        # for p in filter(lambda p: p.grad is not None, self.critic.parameters()):
        #     p.grad.data.clamp_(min=-1, max=1)
        self.critic_opt.step()

        advantage = (target_q - pred_v).detach()
        policy_loss = -advantage * log_prob + 0.01 * log_prob.exp() * log_prob

        self.actor_opt.zero_grad()
        policy_loss.backward()
        # for p in filter(lambda p: p.grad is not None, self.actor.parameters()):
        #     p.grad.data.clamp_(min=-1, max=1)
        self.actor_opt.step()
        # print(policy_loss, critic_loss, policy_loss.item(), critic_loss.item())
        return policy_loss.item(), critic_loss.item()

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            while True:
                if is_render:
                    self.env.render()
                with torch.no_grad():
                    a, _ = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
