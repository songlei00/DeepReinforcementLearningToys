import gym
from td3_algorithm import TD3


env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2']
env_name = env_names[2]
env = gym.make(env_name)
td3 = TD3(env_name, env)
td3.train(10000)
