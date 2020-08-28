# Soft actor-critic (SAC)

The soft actor-critic algorithm is an off-policy maximum entropy deep reinforcement learning algorithm that provides sample-efficient learning while retaining the benefits of entropy maximization and stability. 

[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

[论文作者的代码实现](https://github.com/rail-berkeley/softlearning)

SAC的实现中

第二个版本的SAC相比之前，删除了state value function的估计网络，并且调整$alpha$为学习的参数。