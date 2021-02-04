import gym
from torchvision import transforms
import torch
from torch import nn, optim
from torch.nn import Sequential, Conv2d, ReLU, Linear, Softmax, Flatten
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
import numpy as np

import sys
sys.path.append('..')
from common.utils import Memory
import os

# if torch.backends.cudnn.enabled:
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# seed = 777
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

class Network(nn.Module):

    def __init__(self, output_size):
        super(Network, self).__init__()
        self.net = Sequential(
            Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ReLU(),
            Flatten(),
            Linear(7*7*64, 512),
            ReLU(),
            Linear(512, output_size),
            # Softmax(),
        )
        self.initialize()

    def forward(self, x):
        x = self.net(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)


class DQN:

    def __init__(self,
                 env_name,
                 env,
                 batch_size=32,
                 replay_memory_size=1e6,
                 history_size=4,
                 target_net_update_frequency=1e4,
                 gamma=0.99,
                 action_repeat=4,
                 lr=0.00025,
                 gradient_momentum=0.95,
                 initial_epsilon=1,
                 final_epsilon=0.1,
                 epsilon_decay_step=1e6,
                 warmup_step=5e4,
                 save_model_frequency=20,
                 eval_frequency=1):
        self.env_name = env_name
        self.env = env
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.history_size = history_size
        self.target_net_update_frequency = target_net_update_frequency
        self.gamma = gamma
        self.action_repeat = action_repeat
        self.lr = lr
        self.gradient_momentum = gradient_momentum
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_step = epsilon_decay_step
        self.warmup_step = warmup_step
        self.save_model_frequency = save_model_frequency
        self.eval_frequency = eval_frequency

        self.epsilon = self.initial_epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train on device:', self.device)

        self.memory = Memory(int(replay_memory_size), batch_size)
        self.net = Network(self.env.action_space.n).to(self.device)
        print(self.net)
        self.target_net = Network(self.env.action_space.n).to(self.device)
        self.update_model(self.target_net, self.net)
        self.opt = optim.RMSprop(self.net.parameters(), lr=self.lr, alpha=self.gradient_momentum)
        self.writer = SummaryWriter('./logs/DQN_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.env_name))
        self.loss_fn = F.mse_loss

        self.total_step = 0

    def select_action(self, state, is_test=False):
        epsilon = 0.05 if is_test else self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            a = self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            a = self.net(state).cpu().detach().numpy().argmax()
        return a

    def learn(self):
        batch = self.memory.sample()
        loss = self.compute_loss(batch)
        self.opt.zero_grad()
        loss.backward()
        # for param in self.net.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        for p in filter(lambda p: p.grad is not None, self.net.parameters()):
            p.grad.data.clamp_(min=-1, max=1)

        self.opt.step()

    def compute_loss(self, batch):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = batch.state, batch.action, batch.reward, batch.next_state, batch.done
        batch_state = torch.tensor(batch_state, dtype=torch.float, requires_grad=True).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float, requires_grad=True).to(self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float, requires_grad=True).to(self.device)
        batch_mask = torch.tensor([not i for i in batch_done], dtype=torch.bool).to(self.device)

        pred_q = self.net(batch_state)
        pred_target_q = self.target_net(batch_next_state)
        q = torch.tensor([i[idx] for idx, i in zip(batch_action.long(), pred_q)], dtype=torch.float, requires_grad=True).to(self.device)
        max_next_q = torch.tensor([i.max() for i in pred_target_q], dtype=torch.float, requires_grad=True).to(self.device)
        target_q = batch_mask * (batch_reward + self.gamma * max_next_q)

        loss = self.loss_fn(q, target_q)
        # self.writer.add_scalar('loss', loss, self.total_step)

        return loss

    def train(self, epochs):
        for epoch in range(epochs):
            s = self.env.reset()
            s = self.preprocess(s)
            s = np.stack((s[0], s[0], s[0], s[0]), axis=0)
            while True:
                # self.env.render()

                if self.total_step < self.warmup_step:
                    a = env.action_space.sample()
                else:
                    a = self.select_action(s)

                s_, r, done, _ = env.step(a)
                s_ = self.preprocess(s_)
                s_ = np.stack((s[1], s[2], s[3], s_[0]), axis=0)
                if r > 0:
                    r = 1
                elif r < 0:
                    r = -1
                self.memory.push(s, a, r, s_, done)
                s = s_
                self.total_step += 1
                self.epsilon = self.final_epsilon if self.total_step > self.epsilon_decay_step else self.initial_epsilon - (
                            self.initial_epsilon - self.final_epsilon) * self.total_step / self.epsilon_decay_step

                if self.total_step % self.target_net_update_frequency == 0:
                    self.update_model(self.target_net, self.net)

                if len(self.memory) >= self.batch_size:
                    self.learn()

                if done:
                    break

            if (epoch+1) % self.save_model_frequency == 0:
                self.save_model(self.net, 'model/model_DQN_{}'.format(epoch))

            if (epoch+1) % self.eval_frequency == 0:
                eval_r = self.evaluate()
                print('epoch', epoch, 'reward', eval_r)
                self.writer.add_scalar('reward', eval_r, epoch)

    def preprocess(self, img):
        img = Image.fromarray(img)
        img_preprocess = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((84, 84)),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ])
        img = img_preprocess(img)
        return img.numpy()

    def save_model(self, model, path):
        p = os.path.dirname(path)
        if not os.path.exists(p):
            os.mkdir(p)
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))

    def update_model(self, target_model, model, tau=1):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

    def evaluate(self, epochs=3):
        total_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            s = self.preprocess(s)
            s = np.stack((s[0], s[0], s[0], s[0]), axis=0)
            while True:
                a = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                s_ = self.preprocess(s_)
                s_ = np.stack((s[1], s[2], s[3], s_[0]), axis=0)
                total_r += r
                s = s_
                if done:
                    break

        return total_r / epochs


env_name = ['SpaceInvaders-v0', 'Breakout', 'Pong', 'Tennis', 'Freeway']

env = gym.make(env_name[0])

dqn = DQN(env_name[0], env)
dqn.train(2000)
