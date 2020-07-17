# Deep Q Network

## 1. 简介

DQN将Q learning和神经网络结合，使用神经网络逼近动作价值函数$Q^*(s, a) = max_{\pi}E[r_t + \gamma r_{t+1} + \gamma^2r_{t+2} | s_t=s, a_t=a, \pi]$。这种方法是model free、off policy的。

但由于连续获得的观察具有较强相关性；Q函数的微小更新可能导致策略发生很大变化，最终导致采集到的数据分布的变化；值函数$Q$和目标值$r+\gamma max_{a'}Q(s', a')$具有相关性的原因，使用非线性的近似函数，如神经网络进行强化学习得到的结果往往是不稳定甚至无法收敛的。DQN通过使用experience replay和fixed Q target两种方法解决这个问题。

Experience replay可以打乱观察数据，减弱观察之间相关性和采集的数据分布发生变化产生的影响，并且提高了数据使用率。实现是将每次与环境交互得到的经验$e_t=(s_t, a_t, r_t, s_{t+1})$保存下来，每次训练时从历史中随机选择一个$(s, a, r, s')\sim U(D)$更新参数$\theta$。

Fixed Q target减弱目标值和当前$Q$值之间的相关性。实现是将用于计算目标值$r + \gamma _{a'}Q(s', a', \theta_i^-)$的参数$\theta^-$固定，每隔C轮才更新为Q网络的参数$\theta$。

如果是训练玩Atari游戏，则还有很多处理图像的技巧，见论文。

## 2. 相关论文

Human Level Control Through Deep Reinforement Learning.

