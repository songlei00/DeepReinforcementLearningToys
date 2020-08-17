import gym

env = gym.make('HalfCheetah-v2')

for epoch in range(100):
    s = env.reset()
    tot_r = 0
    i = 0

    while True:
        env.render()
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        i += 1
        tot_r += r

        s = s_
        if done:
            break
    print(i, tot_r)