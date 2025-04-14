

# 维度            强化学习(贝尔曼方程)                    控制理论(HJB方程)
# 模型依赖      无需显式模型(无模型方法)	            依赖精确模型(f(x,u)f(x,u)已知)
# 随机性处理    通过采样和期望直接建模(如Q学习)	         需显式定义扩散过程(如随机微分方程)
# 计算复杂度    高维状态空间需近似(值函数逼近、DQN等)     依赖PDE数值解，维数灾难更严重
# 实时性	    在线学习，适应动态环境	               通常离线求解，需预知系统模型


# 贝尔曼方程与HJB方程本质是同一思想（动态规划）在不同问题（离散/连续、随机/确定）中的数学表达。

# 贝尔曼最优方程是 离散时间 下寻找最优策略的核心工具，而HJB方程是其 连续时间 版本的微分形式。

# 期望方程是贝尔曼方程在随机系统中的具体实现，强调对不确定性的显式建模。

# 在模型已知的简单场景（如LQR），两者可转化为同一类方程（如Riccati方程）；在复杂场景（非线性、高维、无模型），强化学习方法更灵活，而控制理论方法依赖精确建模。



# Riccati与LQR方程参考：
# https://zhuanlan.zhihu.com/p/692283143


import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils as rl_utils


env = gym.make('CartPole-v1')
# | Num | Observation           | Min                 | Max               |
# |-----|-----------------------|---------------------|-------------------|
# | 0   | Cart Position         | -4.8                | 4.8               |
# | 1   | Cart Velocity         | -Inf                | Inf               |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |
### --- ###
# | Num | Action                 |
# |-----|------------------------|
# | 0   | Push cart to the left  |
# | 1   | Push cart to the right |
