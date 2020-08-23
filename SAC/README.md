The soft actor-critic algorithm is an off-policy maximum entropy deep reinforcement learning algorithm that provides sample-efficient learning while retaining the benefits of entropy maximization and stability. The 

Reference:



原本的SAC，

第二个版本的SAC相比之前，删除了state value function的估计网络，并且调整$alpha$为学习的参数。