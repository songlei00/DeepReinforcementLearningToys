import torch


class BaseAgent:

    def __init__(self,
                 env_name,
                 env):
        self.env_name = env_name
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state, is_test=True):
        raise NotImplementedError("select_action: not implemented")

    def train(self, epochs):
        raise NotImplementedError("train: not implemented")

    def learn(self):
        raise NotImplementedError("learn: not implemented")

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
