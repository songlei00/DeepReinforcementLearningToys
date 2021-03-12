import gym

env_name = ['Hopper-v2', 'HalfCheetah-v2']
env = gym.make(env_name[1])

for epoch in range(100):
    s = env.reset()
    eval_r = 0
    data_cnt = 0
    while True:
        env.render()
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        data_cnt += 1
        s = s_
        eval_r += r
        if done:
            break
    print(epoch, eval_r, data_cnt)
