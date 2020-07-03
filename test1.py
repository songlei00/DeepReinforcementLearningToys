import gym
import numpy as np 

env = gym.make('CartPole-v0')

for episode in range(10):
    env.reset()
    while True:
        env.render()
        # a = np.random.randint(0, 2)
        a = int(input())
        obs, r, done, _ = env.step(a)
        print(obs, r, done)
        if done:
            break

env.close()