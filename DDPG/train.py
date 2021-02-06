import gym
from ddpg_algorithm import DDPG


env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2']
env_name = env_names[2]
env = gym.make(env_name)
ddpg = DDPG(env_name, env)
ddpg.train(10000)
