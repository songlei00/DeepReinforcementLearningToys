import gym
from sac_algorithm import SAC


env_names = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[0]
env = gym.make(env_name)
sac = SAC(env_name, env)
sac.train(10000)
