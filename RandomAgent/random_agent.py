import gym


env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)

for epoch in range(100):
    s = env.reset()
    eval_r = 0
    while True:
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        s = s_
        eval_r += r
        if done:
            break
    print(epoch, eval_r)
