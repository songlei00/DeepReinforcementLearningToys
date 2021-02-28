import gym
from trpo_algorithm import TRPO


env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[0]
env = gym.make(env_name)
trpo = TRPO(env_name, env, sample_size=1024)
trpo.train(10000)
