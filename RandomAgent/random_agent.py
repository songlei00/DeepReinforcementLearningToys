import gym

env_name = ['SpaceInvaders-v0', 'Breakout-v0', 'Pong-v0', 'BeamRider-v0', 'MsPacman-v0', 'Pendulum-v0']
env = gym.make(env_name[5])

for epoch in range(100):
    s = env.reset()
    eval_r = 0
    while True:
        env.render()
        a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        s = s_
        eval_r += r
        if done:
            break
    print(epoch, eval_r)
