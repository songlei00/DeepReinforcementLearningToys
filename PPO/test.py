import gym
from ppo_algorithm import PPO

import sys
sys.path.append('..')
from common.utils import load_model, ZFilter


env_names = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']
env_name = env_names[0]
env = gym.make(env_name)
ppo = PPO(env_name, env, is_test=True)

load_model(ppo.actor, 'model/{}_model/best_actor'.format(env_name))
load_model(ppo.critic, 'model/{}_model/best_critic'.format(env_name))
ppo.state_normalize = ZFilter.load('model/{}_model/best_rs'.format(env_name))

for _ in range(10):
    eval_r = ppo.evaluate(1, is_render=True)
    print('evaluate reward', eval_r)
