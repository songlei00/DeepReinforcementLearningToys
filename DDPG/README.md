# DDPG

## 相关论文

[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)

[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

## 算法思路

DDPG是延续DQN的思路

将之前的从策略分布中采样得到一个随机策略的方法转化为确定性的策略。由于确定性策略降低了agent探索的能力，所以引入了off-policy的思想(通过加OU过程的噪声的策略探索，但学习的是确定性的策略)提高样本利用率，使用DQN中的target net并进行soft update和experience replay的方法提高训练的稳定性。

## 实验结果

感觉DDPG效果很好，同样的参数在多个任务上都有不错的效果。

```Pendulum-v0```：效果很好，学得很快(几十轮就可以看到很明显的效果，比A2C强多了...)，但不太稳定。加入Running mean std后，

```HalfCheetah-v2```：效果同样很好，几十轮就可以看到明显效果(每轮个样本，所以大概个样本可以看到效果)，结果也比较稳定，跑动姿势也比较正常。

```Hopper-v2```：效果也不错，姿势也比较正常，但不太稳定。
