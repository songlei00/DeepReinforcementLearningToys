# TRPO

## 相关论文

[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

## 算法思路

这应该是最长的一个```README.md```了，复杂的数学运算。。。

RL的目标是最大化累计奖赏$\eta(\pi)=\mathbb{E_{\tau\sim \pi}[\sum_{t=0}^{\infty}\gamma^t r(s_t)]}$。TRPO主要是从理论上推导出一种累积奖赏$\eta(\pi)$的下界，用这个下界近似原目标，通过优化下界最大化累计奖赏。

由于这种近似只在小范围内成立，所以用置信域限制了参数变化的范围，使得近似足够精确。

所以最终的优化目标是最大化近似的函数，而限制是参数变化范围不能过大。

### 优化问题推导

1. 首先从理论上证明得到一种使得策略单调提升的算法。

首先定义符号
$$Q_{\pi}(s_t, a_t) = \mathbb{E_{s_{t+1}, a_{t+1}, ...}[\sum_{l=0}^{\infty}\gamma^l r(s_{t+l})]}$$
$$V_{\pi}(s_t) = \mathbb{E_{a_t, s_{t+1}, ...}[\sum_{l=0}^{\infty} \gamma^l r(s_{t+l})]}$$
$$A_{\pi} (s, a) = Q_{\pi}(s, a) - V_{\pi}(s)$$

然后证明$$\eta(\widetilde{\pi}) = \eta(\pi) + \mathbb{E}_{\tau\sim \widetilde{\pi}}[\sum_{t=0}^{\infty}\gamma ^t A_{\pi}(s_t, s_t)] = \eta(\pi) + \sum_s\rho_{\widetilde{\pi}}(s) \sum_a \widetilde{\pi}(a|s)A_{\pi}(s, a)$$(证明1)

这个等式的左侧是新的累计奖赏，右侧是旧累计奖赏加$\sum_s\rho_{\widetilde{\pi}}(s) \sum_a \widetilde{\pi}(a|s)A_{\pi}(s, a)$，所以当第二项$\sum_s\rho_{\widetilde{\pi}}(s) \sum_a \widetilde{\pi}(a|s)A_{\pi}(s, a)>0$时，新策略优于旧策略，所以我们的目标就是最大化第二项。

2. 但上式中涉及$\rho_{\widetilde{\pi}}$，$\rho_{\widetilde{\pi}}$每次都需要用新策略$\widetilde{\pi}$采样，不能使用旧的数据，数据利用率极低。所以通过$\rho_{\pi}$近似这一项，使得算法可以重复利用旧数据，提高数据利用率。得到新的式子：

$$L_{\pi}(\widetilde{\pi}) = \eta(\pi) + \mathbb{E}_{\tau\sim \pi}[\sum_{t=0}^{\infty}\gamma ^t A_{\pi}(s_t, s_t)]$$

这种近似原式的一阶近似(证明2)。所以在步长较小时，可以很好的近似原式，但步长较大时，则不一定，所以需要限制步长为一个比较小的值。这时就引入了Trust Region置信域的概念(置信域就是限制更新步长在某个范围内，新策略和旧策略的差别不大，以保证一阶近似足够精确)。通过KL散度衡量两种策略的差异性，差异性要小于某个常数$\delta$，所以得到了优化问题(优化目标是最大化累计奖赏，限制是新旧策略差异不大)如下：

$$max_{\theta} L_{\theta_{old}}(\theta)$$
$$s.t. D_{KL}^{max}(\theta_{old}, \theta) \leq \delta$$

用平均KL代替最大KL得到最终的优化目标，得到的优化问题为

$$max_{\theta} \mathbb{E}_{s\sim \rho_{\theta_{old}}, a\sim q}[\frac{\pi_{\theta}(a|s)}{q(a|s)}Q_{\theta_{old}}(s, a)] = max_{\theta} (\eta(\theta_{old}) + \sum_s\rho_{\pi}(s)\sum_a\widetilde{\pi}(a|s)A_{\pi}(s, a))$$

$$s.t. \mathbb{E}_{s\sim \rho_{\theta_{old}}}[D_{KL}(\pi_{\theta_{old}}(·|s) || \pi_{\theta}(·|s))] \leq \delta$$

第一项是常数，第二项需要对新策略$\widetilde{\pi}(a|s)$采样，所以通过重要性采样转化为对旧策略$q$的采样$\sum_a\widetilde{\pi}(a|s)A_{\theta_{old}}(s, a) = \mathbb{E}_{a\sim q}[\frac{\pi_{\theta}(a|s)}{q(a|s)}A_{\theta_{old}}(s, a)]$

3. 所以最终得到的优化问题为

$$max \mathbb{E}_{s\sim\rho_{\theta_{old}}, a\sim q}[\frac{\pi_{\theta}(a|s)}{q(a|s)}A_{\theta_{old}}(s, a)]$$

$$s.t. \mathbb{E}_{s\sim \rho_{\theta_{old}}}[D_{KL}(\pi_{\theta_{old}}(·|s) || \pi_{\theta}(·|s))] \leq \delta$$

### 求解优化问题

代码实现时就是通过不断求解该优化问题得到最优策略。先利用拉格朗日乘子法、KKT条件，解得结果(最后求得的其实就是自然梯度)。通过利用共轭梯度法得到hessian矩阵的逆和策略梯度的乘积，得到参数更新方向，使用回溯线搜索得到参数的更新步长。

参考：https://www.yuque.com/linke-dxzuq/rl/lgt4py

### 其他

TRPO是on policy的方法，on policy因为只利用当前数据，所以这类方法重心更多在提高数据利用率上，利用TRPO通过取目标函数的下界，重复利用原数据，提高利用率。另一类的off policy的方法由于使用了replay buffer，重复利用历史数据，所以数据利用率更高，但同时也带来了不稳定性，所以off policy的方法更加关注如何使得算法更加稳定，例如使用target net，delayed update等等。(这段话也许是错的)

## 实验结果

### HalfCheetah-v2
