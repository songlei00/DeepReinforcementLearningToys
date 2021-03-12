# PPO

## 相关论文

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，openai提出的PPO算法

[OpenAI blog](https://openai.com/blog/openai-baselines-ppo/)

[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)，deepmind提出的Distributed PPO，加入了分布式计算

[Implementation Matters in Deep Policy Gradients: a Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729v1)，对PPO和TRPO的比较，认为PPO性能更好是由于代码实现更好(这个论文里有很多PPO代码实现的trick，这些trick并没有在论文中提及)

[Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms](https://arxiv.org/abs/1811.02553v2)

[Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)

[OpenAI baselines中PPO1和PPO2的实现](https://github.com/openai/baselines)

[一个我感觉很好的实现](https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py)

对PPO的吐槽感觉很多，但也架不住PPO的效果实在太好了。。。

## 算法思路

学习PPO之前建议先看TRPO的推导部分

PPO实现的trick是真的多。。。

使用batch的数据多次更新，熵正则化，GAE，状态归一化和clip，值函数clip，奖赏归一化或者clip，正交初始化

PPO和PPO2，PPO是使用的KL作为惩罚项并自适应的$\beta$的版本，PPO2则是使用clip限制

### 实现过程中的亿点点小问题

由于多步更新，所以必须用clip的trick，否则会更新错误很多

#### PPO计算loss时

这对于PPO非常重要，因为如果一次只更新1次，那么就是完全的on policy，ratio=1，ppo完全没用到clip，policy训练会非常慢。因此，最理想的是更新多次，更新到clip的边缘，最大化每一次采集样本的利用。

同时，基于上面的设定，actor/worker越多，一次采集的样本就越多，在控制训练次数的情况下，batchsize就会越来越大。对于PPO，batch 越大，对梯度的估计就越准，bias越小，所以效果会越好。在OpenAI Dota Five中，采用了极大的batchsize来加速训练。这也是large scale最大的意义。


clip后梯度为什么为0

还有个问题，ppo是同时优化actor和critic的，但这两个loss差距过大，这怎么能优化出来呢(必须要做value的norm吗)

因为重要的是梯度吗，和loss的值关系不大？？？

std使用参数和从网络中得到有什么区别

似乎作为一个可训练参数而不是网络输出会增加鲁棒性

actor梯度的问题

DDPG中的actor更新是使用actor计算action，然后传入critic中，将critic作为网络的后半部分来计算梯度，

DDPG也训练一个 Critic Network 去估计state-action的Q值，然后把Critic Network“连在”Actor Network的后面，让Critic 为策略网络Actor 提供优化的梯度

## 实验结果
