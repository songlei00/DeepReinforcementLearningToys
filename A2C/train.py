import gym
import math
# from a2c_algorithm_step_update import A2C
from a2c_algorithm_epoch_update import A2C

# 'Pendulum-v0'
env_name = 'Pendulum-v0'
env = gym.make(env_name)
a2c = A2C(env_name, env, is_continue_action_space=True)
a2c.train(3000)


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
