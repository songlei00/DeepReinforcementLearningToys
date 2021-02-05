import gym
from ddpg_algorithm import DDPG

import sys
sys.path.append('..')
from common.utils import load_model


env_names = ['Pendulum-v0', ]
env_name = env_names[0]
env = gym.make(env_name)
ddpg = DDPG(env_name, env, is_test=True)

load_model(ddpg.actor, 'model/{}_model/best_actor'.format(env_name))
load_model(ddpg.critic, 'model/{}_model/best_critic'.format(env_name))
for _ in range(10):
    eval_r = ddpg.evaluate(5, is_render=True)
    print('evaluate reward', eval_r)
