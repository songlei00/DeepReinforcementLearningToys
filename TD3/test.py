import gym
from td3_algorithm import TD3

import sys
sys.path.append('..')
from common.utils import load_model, ZFilter


env_names = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[2]
env = gym.make(env_name)
td3 = TD3(env_name, env, is_test=True)

load_model(td3.actor, 'model/{}_model/best_actor'.format(env_name))
load_model(td3.critic, 'model/{}_model/best_critic'.format(env_name))
td3.state_normalize = ZFilter.load('model/{}_model/best_rs'.format(env_name))

for _ in range(10):
    eval_r = td3.evaluate(1, is_render=True)
    print('evaluate reward', eval_r)
