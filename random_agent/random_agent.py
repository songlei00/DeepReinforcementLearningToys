import sys
sys.path.append('../tools')
from agent import RandomAgent
import gym


if __name__ == '__main__':
    env = gym.make('Hopper-v2')
    agent = RandomAgent('Random', env)
    agent.train(1000)
    agent.plot_reward()