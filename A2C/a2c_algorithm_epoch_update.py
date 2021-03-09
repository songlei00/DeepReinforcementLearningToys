import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append('..')
from common.network import GaussianActor, Critic
from common.utils import save_model, ZFilter, Trace


class A2C:

    def __init__(self,
                 env_name,
                 env,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 sample_size=2048,
                 gamma=0.99,
                 lam=0.95,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=10):
        self.env_name = env_name
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.sample_size = sample_size
        self.gamma = gamma
        self.lam = lam
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.total_step = 0
        self.state_normalize = ZFilter(env.observation_space.shape[0])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs_epoch_update/A2C_{}'.format(self.env_name))
        self.loss_fn = F.smooth_l1_loss

        self.trace = Trace()
        self.actor = GaussianActor(env.observation_space.shape[0], env.action_space.shape[0],
                                   action_scale=int(env.action_space.high[0])).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = Critic(env.observation_space.shape[0]).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        print(self.actor)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        a, log_prob = self.actor.sample(state, is_test)
        return a, log_prob

    def train(self, epochs):
        best_eval = -1e6
        for epoch in range(epochs):
            num_sample = 0
            self.trace.clear()
            s = self.env.reset()
            s = self.state_normalize(s)
            while num_sample < self.sample_size:
                self.env.render()
                a, log_prob = self.select_action(s)
                torch_s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
                v = self.critic(torch_s)
                s_, r, done, _ = self.env.step(a.cpu().detach().numpy()[0])
                s_ = self.state_normalize(s_)
                self.trace.push(s, a, log_prob, r, s_, not done, v)  # 这里怎么写才能在learn里面不用reshape呢
                num_sample += 1
                self.total_step += 1
                s = s_
                if done:
                    s = self.env.reset()
                    s = self.state_normalize(s)

            policy_loss, critic_loss = self.learn()

            self.writer.add_scalar('loss/actor_loss', policy_loss, self.total_step)
            self.writer.add_scalar('loss/critic_loss', critic_loss, self.total_step)

            if (epoch + 1) % self.save_model_frequency == 0:
                save_model(self.critic, 'model_epoch_update/{}_model/critic_{}'.format(self.env_name, self.total_step))
                save_model(self.actor, 'model_epoch_update/{}_model/actor_{}'.format(self.env_name, self.total_step))

            if (epoch + 1) % self.eval_frequency == 0:
                eval_r = self.evaluate()
                print('epoch', epoch, 'evaluate reward', eval_r)
                self.writer.add_scalar('reward', eval_r, epoch)
                if eval_r > best_eval:
                    best_eval = eval_r
                    save_model(self.critic, 'model_epoch_update/{}_model/best_critic'.format(self.env_name))
                    save_model(self.actor, 'model_epoch_update/{}_model/best_actor'.format(self.env_name))

    def learn(self):
        all_data = self.trace.get()
        all_state = torch.tensor(all_data.state, dtype=torch.float).to(self.device)
        all_log_prob = torch.cat(all_data.log_prob).to(self.device)

        adv, total_reward = self.trace.cal_advantage(self.gamma, self.lam)
        adv = adv.reshape(len(self.trace), -1).to(self.device)
        total_reward = total_reward.reshape(len(self.trace), -1).to(self.device)

        all_value = self.critic(all_state)
        critic_loss = self.loss_fn(all_value, total_reward.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        policy_loss = (- all_log_prob * adv.detach() + 0.01 * all_log_prob.exp() * all_log_prob).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        return policy_loss.item(), critic_loss.item()

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            s = self.state_normalize(s, update=False)
            while True:
                if is_render:
                    self.env.render()
                with torch.no_grad():
                    a, _ = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a.cpu().detach().numpy()[0])
                s_ = self.state_normalize(s_, update=False)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
