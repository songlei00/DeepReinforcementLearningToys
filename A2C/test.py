import gym
from a2c_algorithm_step_update import A2C

import sys
sys.path.append('..')
from common.utils import load_model


# env_name = 'Pendulum-v0'
# env = gym.make(env_name)
# a2c = A2C(env_name, env, is_continue_action_space=True)

env_name = 'MountainCar-v0'
env = gym.make(env_name)
a2c = A2C(env_name, env, is_continue_action_space=False, is_test=True)

load_model(a2c.actor, 'model_step_update/{}_model/best_actor'.format(env_name))
load_model(a2c.critic, 'model_step_update/{}_model/best_critic'.format(env_name))
for _ in range(10):
    eval_r = a2c.evaluate(5, is_render=True)
    print('evaluate reward', eval_r)

