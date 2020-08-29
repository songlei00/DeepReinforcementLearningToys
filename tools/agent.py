from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import shutil


class BaseAgent:
    """A base agent that realizes universal method for a complex agent

    Some method to load, save and update model, evaluate the algorithm and 
    so on are provided.

    The user should implement select_action and train in the subclass.
    """

    def __init__(self):
        # auto-set parameters
        self.global_epoch = 0
        self.global_step = 0
        self.reward_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG_PATH = './log'
        shutil.rmtree(LOG_PATH, ignore_errors=True)
        self.writer = SummaryWriter(LOG_PATH)

    def select_action(self, state, is_evaluate=False):
        raise NotImplementedError

    def train(self, epochs):
        raise NotImplementedError

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state(torch.load(path))

    def update_target(self, target_net, net, tau=1):
        """update target network, tau=1 means hard update

        target = (1 - tau) * target + tau * current_network
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1-tau) * target_param.data + tau * param.data)

    def evaluate(self, epochs=1):
        total_reward = 0
        for _ in range(epochs):
            s = self.env.reset()

            while True:
                a = self.select_action(s, is_evaluate=True)
                s_, r, done, _ = self.env.step(a)
                total_reward += r
                if done:
                    break
                s = s_

        total_reward /= epochs 
        self.reward_list.append(total_reward)
        return total_reward

    def plot_reward(self, is_save=True, is_show=True):
        plt.plot(self.reward_list)
        plt.title(self.env_name)
        plt.xlabel('epoch')
        plt.ylabel('reward')
        if is_save:
            plt.savefig(self.env_name + '.png')
        if is_show:
            plt.show()


class RandomAgent(BaseAgent):

    def __init__(
        self,
        env_name,
        env,
        evaluate_interval=10
    ):
        BaseAgent.__init__(self)
        self.env_name = env_name
        self.env = env
        self.evaluate_interval = evaluate_interval

    def select_action(self, state, is_evaluate=False):
        a = self.env.action_space.sample() 
        return a

    def train(self, epochs):
        for _ in range(epochs):
            s = self.env.reset()

            while True:
                self.env.render()
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                self.global_step += 1
                if done:
                    break
                s = s_

            self.global_epoch += 1

            if self.global_epoch % self.evaluate_interval == 0:
                eval_r = self.evaluate()
                self.writer.add_scalar('reward/random_agent_total_reward', eval_r, self.global_epoch)
                print('epoch:', self.global_epoch, 'reward:', eval_r)

