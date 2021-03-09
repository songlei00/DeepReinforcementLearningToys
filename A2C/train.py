import gym
import math
from a2c_algorithm_epoch_update import A2C

env_names = ['Pendulum-v0', 'HalfCheetah-v2']
env_name = env_names[0]
env = gym.make(env_name)
a2c = A2C(env_name, env)
a2c.train(10000)


# from a2c_algorithm_step_update import A2C

# env_names = ['Pendulum-v0', 'HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
# env_name = env_names[1]
# env = gym.make(env_name)
# a2c = A2C(env_name, env, is_continue_action_space=True)
# a2c.train(10000)


# 'MountainCar-v0'
# def mountain_car_reward_func(x):
#     s_, r, done, info = x
#     return math.e**(s_[0]*10 + s_[1])
#
#
# env_name = 'MountainCar-v0'
# env = gym.make(env_name)
# a2c = A2C(env_name, env, reward_shapeing_func=mountain_car_reward_func, is_continue_action_space=False)
# a2c.train(7000)
