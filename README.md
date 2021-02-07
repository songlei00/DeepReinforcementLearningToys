# Reinforcement Learning

## 1. 算法

1. A2C
2. DDPG

## 2. 实验结果

### Logs

记录在对应算法的```logs```文件夹对应环境中，进入目录使用命令```tensorboard --logdir=.```查看。

### 模型

运行算法下的```test.py```文件测试。

## 3. Trick

1. Target net，DQN，DDPG
2. Experience replay，DQN，DDPG
3. Action repeat，在Atari的游戏中，重复执行多次当前动作。DQN，DDPG
4. Actor和critic的权重共享，在Arari游戏中，共享卷积层的权重
5. Soft update，$\theta'=\tau \theta + (1-\tau)\theta', \tau << 1$，使得目标值更加稳定，学习过程接近于监督学习。DDPG
6. 探索过程加入随机噪声。DDPG中加OU
7. Running mean std，利用当前采集到的所有状态对state进行归一化，能够让训练更加稳定(但也增大了计算开销，运行速度会变慢)，见```common/utils.py:ZFilter```。所有的算法都可以用
