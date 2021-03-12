import gym
from ppo_algorithm import PPO


env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[0]
env = gym.make(env_name)
ppo = PPO(env_name, env)
ppo.train(10000)
