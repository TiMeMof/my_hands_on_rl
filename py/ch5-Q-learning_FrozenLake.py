import gym
import time
import numpy as np
import tqdm
import random

# https://zhuanlan.zhihu.com/p/670784628

# 先是不打滑版本（打滑版本有一定概率使得动作失效）
env = gym.make("FrozenLake-v1",render_mode="rgb_array", is_slippery=False)  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.reset()
env.render()
