# Reinforcement Learning

## 算法

1. A2C，on-policy
2. DDPG，off-policy

## 实验结果

记录在对应算法的```logs```文件夹下，使用命令```tensorboard --logdir=.```查看。

## Trick

1. Target net，DQN，DDPG
2. Experience replay，DQN，DDPG
3. Action repeat，DQN，DDPG
3. Soft update，DDPG
4. 探索过程加入随机噪声，DDPG中加OU
5. Running mean std，在RL中通常不使用BN层，但会对状态进行归一化
