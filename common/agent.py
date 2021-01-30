import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class BaseAgent:

    def __init__(self,
                 env_name,
                 env):
        self.env_name = env_name
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter('./logs/{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.env_name))

    def train(self, epochs):
        raise NotImplementedError("train: not implemented")

    @staticmethod
    def save_model(model, path):
        p = os.path.dirname(path)
        if not os.path.exists(p):
            os.mkdir(p)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(model, path):
        model.load_state_dict(torch.load(path))

    @staticmethod
    def update_model(target_model, model, tau=1):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

    def evaluate(self, epochs=1):
        total_r = 0
        for _ in range(epochs):
            s = self.env.reset()
            while True:
                a = self.select_action(s, is_test=True)
                s_, r, done, _ = self.env.step(a)
                total_r += r
                s = s_
                if done:
                    break

        return total_r / epochs


