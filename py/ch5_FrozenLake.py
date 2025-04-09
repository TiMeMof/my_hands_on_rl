import gym
import time
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# https://zhuanlan.zhihu.com/p/670784628

class QLearningAgent:
    def __init__(self, env: gym.Env, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.99,epsilon_decay = 0.001):
        self.env = env
        self.env.reset()
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate # alpha,学习率
        self.discount_factor = discount_factor # gamma,折扣因子
        self.exploration_prob = exploration_prob # epsilon-贪婪策略中的epsilon
        self.exploration_decay = exploration_decay
        print(f"observation_space: {env.observation_space.n}, action_space: {env.action_space.n}")
        # 初始化 Q 表
        # Q 表的维度是 (状态空间大小, 动作空间大小)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return self.env.action_space.sample()  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用
        
    def update_q_table(self, state, action, reward, next_state):
        # Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        # 计算Temporal Difference,时序差分
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# Sarsa算法
class Sarsa:
    def __init__(self, env: gym.Env, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.001):
        self.env = env
        self.env.reset()
        self.learning_rate = learning_rate # alpha,学习率
        self.discount_factor = discount_factor # gamma,折扣因子
        self.exploration_prob = exploration_prob # epsilon-贪婪策略中的epsilon
        self.exploration_decay = exploration_decay
        print(f"observation_space: {env.observation_space.n}, action_space: {env.action_space.n}")
        # 初始化 Q 表
        # Q 表的维度是 (状态空间大小, 动作空间大小)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    def update_q_table(self, state, action, reward, next_state, next_action):
        # Q(s, a) = Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a))
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# 可以设置打滑/不打滑版本（打滑版本有一定概率使得动作失效）
slippery = True
epsilon = 1#设置的越小，越难训练
alpha = 0.1
gamma = 0.95
num_episodes = 10000
epsilon_decay = 0.001
env = gym.make("FrozenLake-v1",render_mode="rgb_array", is_slippery=slippery)  # 创建环境

agent = QLearningAgent(env, learning_rate=alpha, discount_factor=gamma, exploration_prob=epsilon, epsilon_decay=epsilon_decay)
return_list = []
with tqdm(total=int(num_episodes ), desc='Iteration') as pbar:
    for episode in range(num_episodes):
        episode_return = 0
        state, _ = env.reset()
        # env.render()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_return += reward
            agent.update_q_table(state, action, reward, next_state)
            # print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
            state = next_state
        # # 衰减探索概率
        agent.exploration_prob = max(agent.exploration_prob - agent.epsilon_decay, 0)

        return_list.append(episode_return)
        pbar.update(1)

# 0左，1下，2右，3上
env.close()

# 训练sarsa
env2 = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=slippery)  # 创建环境
agent2 = Sarsa(env2, learning_rate=alpha, discount_factor=gamma, exploration_prob=epsilon, exploration_decay=epsilon_decay)
return_list2 = []
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
    for episode in range(num_episodes):
        episode_return = 0
        state, _ = env2.reset()
        action = agent2.choose_action(state)
        done = False
        while not done:
            next_state, reward, done, _, _ = env2.step(action)
            episode_return += reward
            next_action = agent2.choose_action(next_state)
            agent2.update_q_table(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        # 衰减探索概率
        agent2.exploration_prob = max(agent2.exploration_prob - agent2.exploration_decay, 0)

        return_list2.append(episode_return)
        pbar.update(1)

env2.close()

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list, label='Q-learning')
# plt.plot(episodes_list, return_list2, label='Sarsa')
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Training Returns')
# plt.show()

print("Q 表：")
print("Q-learning:\n",agent.q_table,"\n","Sarsa:\n",agent2.q_table)
# 测试训练好的 Q 表
Q = agent2.q_table
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=slippery)
# env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=slippery)

total_cnt = 10000
total_reward = 0
for i in range(total_cnt):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        # print(f"Action: {action}, State: {state}, Reward: {reward}, Q-Value: {Q[state]}")
        # time.sleep(0.2)

print(f"success rate: {total_reward / total_cnt * 100:.2f}%")
env.close()
