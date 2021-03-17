import gym
from sac_algorithm import SAC


env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[2]
env = gym.make(env_name)
# sac = SAC(env_name, env)
sac = SAC(env_name, env, actor_lr=3e-5, critic_lr=3e-5, alpha_lr=3e-5)
sac.train(10000)
