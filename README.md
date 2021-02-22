# Reinforcement Learning

## 1. 算法

- [x] A2C
- [ ] ACER
- [ ] ACKTR
- [x] DDPG
- [x] TD3
- [ ] TRPO
- [ ] PPO
- [ ] SAC

## 2. 代码结构

```
TODO
```

## 3. 实验结果

### Logs

记录在对应算法的```logs```文件夹对应环境中，进入目录使用命令```tensorboard --logdir=.```查看。或查看对应算法目录下的```README.md```文件。

### 模型

运行算法下的```test.py```文件测试，模型是在GPU上训练出来的，所以如果在CPU上运行，需要做一定的修改。

## 4. Trick

1. Target net，terget net通过提供一个稳定的更新方向，使得算法更加稳定。DQN，DDPG
2. Experience replay，DQN，DDPG
3. Action repeat，在Atari的游戏中，重复执行多次当前动作。DQN，DDPG
4. Actor和critic的权重共享，在Arari游戏中，共享卷积层的权重
5. Soft update，$\theta'=\tau \theta + (1-\tau)\theta', \tau << 1$，使得目标值更加稳定，学习过程接近于监督学习。DDPG，TD3
6. Deterministic policy加入随机噪声，促进探索。DDPG中加OU噪声，TD3中加高斯噪声
7. Running mean std，online地利用当前采集到的所有状态对state进行归一化，能够让训练更加稳定(但也增大了计算开销，运行速度会变慢)，见```common/utils.py:ZFilter```。所有的算法都可以用(有一定的效果。同时需要注意的是训练时要记录当前的state的平均值，在测试时使用，而不是在测试时重新收集数据，计算平均值)
8. Twin Q network，$y = r + \gamma \min (Q_1(s', \pi(s')), Q_2(s', \pi(s')))$，缓解Q值估计过高的问题。TD3
9. Delayed update，减缓更新actor，target critic和target actor。TD3

## 5. TODO

实现的代码比较简单体现了算法本身的思想，但不利于修改。以后最好可以将策略网络、值网络、经验池等统一借口作为参数传递给算法；加入n steps、随机种子、状态是否归一化、奖赏是否reshape等参数，利于比较使用不同网络、使用不同的经验池等对最终结果的影响；log中记录Q值、动作的概率等更多的数据。
