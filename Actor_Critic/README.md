# Advantage Actor Critic

## 1. 简介

在将RL和DNN结合之后，因为数据分布容易改变和数据相关性大等问题，造成网络很难训练。一种解决方法是使用replay buffer，但这会导致方法变为off policy。A3C则是一种on policy的训练网络的方法，通过并行多个agent使用不同策略收集数据，用asynchronous替代了experience replay的作用。

## 2. 相关论文

[OpenAI Baselines: ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)

## 3. 运行结果