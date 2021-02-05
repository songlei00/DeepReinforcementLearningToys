# DDPG

## 相关论文

[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)

[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

## 算法思路

将之前的从策略分布中采样得到一个随机策略的方法转化为确定性的策略。由于确定性策略降低了agent探索的能力，所以引入了off-policy的思想(通过加OU过程的噪声的策略探索，但学习的是确定性的策略)提高样本利用率，使用DQN中的target net并进行soft update和experience replay的方法提高训练的稳定性。

(感觉DDPG比前面的算法忽然变强了很多)

## 实验结果

```Pendulum-v0```：效果很好，学得很快(几十轮就可以看到很明显的效果，比A2C强多了...)，但不太稳定。加入Running mean std后，
