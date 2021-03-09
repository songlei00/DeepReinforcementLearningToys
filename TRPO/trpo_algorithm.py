import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append('..')
from common.network import GaussianActor, Critic
from common.utils import Trace, update_model, save_model


class TRPO:

    def __init__(self,
                 env_name,
                 env,
                 critic_lr=3e-4,
                 train_iters=20,
                 backtrack_coeff=1,
                 backtrack_damp_coeff=0.5,
                 backtrack_alpha=0.5,
                 delta=0.01,
                 sample_size=2048,
                 gamma=0.99,
                 lam=0.97,
                 is_test=False,
                 save_model_frequency=200,
                 eval_frequency=20):
        self.env_name = env_name
        self.env = env
        self.critic_lr = critic_lr
        self.train_iters = train_iters
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_damp_coeff = backtrack_damp_coeff
        self.backtrack_alpha = backtrack_alpha
        self.sample_size = sample_size
        self.delta = delta
        self.gamma = gamma
        self.lam = lam
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.total_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print('Train on device:', self.device)
        if not is_test:
            self.writer = SummaryWriter('./logs/TD3_{}'.format(self.env_name))
        self.loss_fn = F.mse_loss

        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0]
        self.old_policy = GaussianActor(n_state, n_action, 128, action_scale=int(env.action_space.high[0])).to(self.device)
        self.new_policy = GaussianActor(n_state, n_action, 128, action_scale=int(env.action_space.high[0])).to(self.device)
        update_model(self.old_policy, self.new_policy)
        self.critic = Critic(n_state, 128).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.trace = Trace()

        print(self.new_policy)
        print(self.critic)

    def select_action(self, state, is_test=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        a, _ = self.new_policy.sample(state, is_test)
        return a.cpu().detach().numpy()

    def train(self, epochs):
        best_eval = - 1e6
        for epoch in range(epochs):
            self.trace.clear()
            num_sample = 0
            s = self.env.reset()
            # collect data
            while num_sample < self.sample_size:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                v = self.critic(torch.tensor(s, dtype=torch.float).to(self.device))
                self.trace.push(s, a, r, s_, done, v)
                num_sample += 1

                s = s_
                if done:
                    s = self.env.reset()

            self.learn()
            if (epoch + 1) % self.eval_frequency == 0:
                eval_r = self.evaluate()
                print('epoch', epoch, 'evaluate reward', eval_r)
                self.writer.add_scalar('reward', eval_r, self.total_step)
                # if eval_r > best_eval:
                #     best_eval = eval_r
                #     save_model(self.critic, 'model/{}_model/best_critic'.format(self.env_name))
                #     save_model(self.actor, 'model/{}_model/best_actor'.format(self.env_name))
                #     ZFilter.save(self.state_normalize, 'model/{}_model/best_rs'.format(self.env_name))

    def learn(self):
        state, action, reward, next_state, done, value = self.trace.get()
        advantage, total_reward = self.trace.cal_advantage(self.gamma, self.lam)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        advantage = torch.tensor(advantage, dtype=torch.float).to(self.device)
        total_reward = torch.tensor(total_reward, dtype=torch.float).to(self.device)

        # update critic
        for _ in range(self.train_iters):
            value = self.critic(state).squeeze(1)
            critic_loss = self.loss_fn(value, total_reward)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # update policy
        log_prob_old = self.new_policy.get_log_prob(state, action)
        log_prob_new = self.new_policy.get_log_prob(state, action)
        ratio_old = torch.exp(log_prob_new - log_prob_old.detach())
        policy_loss_old = (ratio_old * advantage).mean()

        gradient = torch.autograd.grad(policy_loss_old, self.new_policy.parameters())
        gradient = TRPO.flatten_tuple(gradient)

        x = self.cg(state, gradient)
        gHg = (self.get_hessian_dot_vec(state, x) * x).sum(0)
        step_size = torch.sqrt(2 * self.delta / (gHg + 1e-8))
        old_params = self.flatten_tuple(self.new_policy.parameters())
        update_model(self.old_policy, self.new_policy)

        # backtracking line search
        expected_improve = (gradient * step_size * x).sum()
        print(expected_improve)
        tmp_backtrack_coeff = self.backtrack_coeff
        for _ in range(self.train_iters):
            new_params = old_params + self.backtrack_coeff * step_size * x
            idx = 0
            for param in self.new_policy.parameters():
                param_len = len(param.view(-1))
                new_param = new_params[idx: idx + param_len]
                new_param = new_param.view(param.size())
                param.data.copy_(new_param)
                idx += param_len

            log_porb = self.new_policy.get_log_prob(state, action)
            ratio = torch.exp(log_porb - log_prob_old)
            policy_loss = (ratio * advantage).mean()
            loss_improve = policy_loss - policy_loss_old
            expected_improve *= tmp_backtrack_coeff
            imporve_condition = (loss_improve / (expected_improve + 1e-8)).item()

            kl = (self.kl_divergence(self.old_policy, self.new_policy, state)).mean()

            if kl < self.delta and imporve_condition > self.backtrack_alpha:
                break

            tmp_backtrack_coeff *= self.backtrack_damp_coeff

    def cg(self, state, g, n_iters=20):
        # conjugate gradient algorithm to solve linear equation: Hx=g, H is symmetric and positive-definite
        # repeat:
        #   alpha_k = (r_k.T * r_k) / (p_k.T * A * p_k)
        #   x_k+1 = x_k + alpha_k * p_k
        #   r_k+1 = r_k + alpha_k * A * p_k
        #   beta_k+1 = (r_k+1.T *  r_k+1) / (r_k.T * r_k)
        #   p_k+1 = -r_k+1 + beta_k+1 * p_k
        #   k = k + 1
        # end repeat
        x = torch.zeros_like(g).to(self.device)
        r = g.clone().to(self.device)
        p = g.clone().to(self.device)
        rdotr = torch.dot(r, r).to(self.device)

        for _ in range(n_iters):
            Hp = self.get_hessian_dot_vec(state, p)
            alpha = rdotr / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r += alpha * Hp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def kl_divergence(self, old_policy, new_policy, state):
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # KL(p, q) = log(sigma_2 / sigma_1) + (sigma_1^2 + (mu_1 - mu_2)^2) / (2*sigma_2^2) - 0.5
        state = torch.as_tensor(state, dtype=torch.float).to(self.device)

        mu_1, log_sigma_1 = old_policy(state)
        mu_1 = mu_1.detach()
        sigma_1 = log_sigma_1.exp().detach()

        mu_2, log_sigma_2 = new_policy(state)
        sigma_2 = log_sigma_2.exp()
        kl = torch.log(sigma_2/sigma_1) + (sigma_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2*sigma_2.pow(2) + 1e-8) - 0.5

        return kl

    def get_hessian_dot_vec(self, state, vec, damping_coeff=0.01):
        kl = self.kl_divergence(self.old_policy, self.new_policy, state)
        kl_mean = kl.mean()
        gradient = torch.autograd.grad(kl_mean, self.new_policy.parameters(), create_graph=True)
        gradient = self.flatten_tuple(gradient)

        kl_grad_p = (gradient * vec).sum()
        kl_hessian = torch.autograd.grad(kl_grad_p, self.new_policy.parameters())
        kl_hessian = self.flatten_tuple(kl_hessian)

        return kl_hessian + damping_coeff * vec

    @staticmethod
    def flatten_tuple(t):
        flatten_t = torch.cat([data.view(-1) for data in t])
        return flatten_t

    def evaluate(self, epochs=3, is_render=False):
        eval_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            # s = self.state_normalize(s, update=False)
            while True:
                if is_render:
                    self.env.render()
                a = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                # s_ = self.state_normalize(s_, update=False)
                s = s_
                eval_r += r
                if done:
                    break
        return eval_r / epochs
