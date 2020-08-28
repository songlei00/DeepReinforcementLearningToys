from tools.agent import RandomAgent
import gym

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = RandomAgent('Random', env)
    agent.train(1000)
    agent.plot_reward()