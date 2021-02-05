import gym
from ddpg_algorithm import DDPG


env_name = 'Pendulum-v0'
env = gym.make(env_name)
ddpg = DDPG(env_name, env)
ddpg.train(7000)


# env_name = 'HalfCheetah-v2'
# env = gym.make(env_name)
# ddpg = DDPG(env_name, env)
# ddpg.train(10000)
