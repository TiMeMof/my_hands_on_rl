import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    

# epsilons = [0.01, 0.01, 0.01] 
epsilons = [0.1, 0.1, 0.1] 
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = epsilons[0] 
alpha = 0.1
gamma = 0.9
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list1 = []  # 记录每一条序列的回报
# 这里改成直接500次迭代是一样的
with tqdm(total=int(num_episodes ), desc='Iteration' ) as pbar:
    for i_episode in range(int(num_episodes )):  # 每个进度条的序列数
        episode_return = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.take_action(next_state)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        return_list1.append(episode_return)
        if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
            pbar.set_postfix({
                'episode':
                '%d' % (i_episode + 1),
                'return':
                '%.3f' % np.mean(return_list1[-10:])
            })
        pbar.update(1)

episodes_list = list(range(len(return_list1)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Sarsa on {}'.format('Cliff Walking'))
# plt.show()

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# Sarsa算法最终收敛得到的策略为：
# ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo
# ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo
# ^ooo ooo> ^ooo ooo> ooo> ooo> ooo> ^ooo ^ooo ooo> ooo> ovoo
# ^ooo **** **** **** **** **** **** **** **** **** **** EEEE


print("\n"+"""### 多步Sarsa算法 ###"""+"\n")

class nstep_Sarsa:
    """ n步Sarsa算法 """
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n  # 采用n步Sarsa算法
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n步Sarsa的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []


np.random.seed(0)
n_step = 5  # 5步Sarsa算法
alpha = 0.1
epsilon = epsilons[1]
gamma = 0.9
agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list2 = []  # 记录每一条序列的回报
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
    for i_episode in range(int(num_episodes)):  # 每个进度条的序列数
        episode_return = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.take_action(next_state)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            agent.update(state, action, reward, next_state, next_action,
                            done)
            state = next_state
            action = next_action
        return_list2.append(episode_return)
        if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
            pbar.set_postfix({
                'episode':
                '%d' % (i_episode + 1),
                'return':
                '%.3f' % np.mean(return_list2[-10:])
            })
        pbar.update(1)

episodes_list = list(range(len(return_list2)))
# plt.plot(episodes_list, return_list2)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
# plt.show()

action_meaning = ['^', 'v', '<', '>']
print('5步Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])





# 5步Sarsa算法最终收敛得到的策略为：
# ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo
# ^ooo ^ooo ^ooo oo<o ^ooo ^ooo ^ooo ^ooo ooo> ooo> ^ooo ovoo
# ooo> ^ooo ^ooo ^ooo ^ooo ^ooo ^ooo ooo> ooo> ^ooo ooo> ovoo
# ^ooo **** **** **** **** **** **** **** **** **** **** EEEE

print("\n"+"""### Q-learning算法 ###"""+"\n")

class QLearning:
    """ Q-learning算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  #选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


np.random.seed(0)
epsilon = epsilons[2]
alpha = 0.1
gamma = 0.9
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list3 = []  # 记录每一条序列的回报
with tqdm(total=int(num_episodes ), desc='Iteration') as pbar:
    for i_episode in range(int(num_episodes )):  # 每个进度条的序列数
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            agent.update(state, action, reward, next_state)
            state = next_state
        return_list3.append(episode_return)
        if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
            pbar.set_postfix({
                'episode':
                '%d' % ( i_episode + 1),
                'return':
                '%.3f' % np.mean(return_list3[-10:])
            })
        pbar.update(1)

return_list4 = []  # 记录每一条序列的回报
with tqdm(total=int(num_episodes ), desc='Iteration') as pbar:
    agent.epsilon = 0
    for i_episode in range(int(num_episodes )):  # 每个进度条的序列数
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            agent.update(state, action, reward, next_state)
            state = next_state
        return_list4.append(episode_return)
        if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
            pbar.set_postfix({
                'episode':
                '%d' % ( i_episode + 1),
                'return':
                '%.3f' % np.mean(return_list4[-10:])
            })
        pbar.update(1)

episodes_list = list(range(len(return_list3)))

action_meaning = ['^', 'v', '<', '>']
print('Q-learning算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
# Q-learning算法最终收敛得到的策略为：
# ^ooo ovoo ovoo ^ooo ^ooo ovoo ooo> ^ooo ^ooo ooo> ooo> ovoo
# ooo> ooo> ooo> ooo> ooo> ooo> ^ooo ooo> ooo> ooo> ooo> ovoo
# ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo
# ^ooo **** **** **** **** **** **** **** **** **** **** EEEE
Qtable = agent.Q_table

plt.figure(figsize=(12, 6))
# # 画散点图
# plt.scatter(episodes_list, return_list1, s=0.5, label='Sarsa')
# plt.scatter(episodes_list, return_list2, s=0.5, label='5-step Sarsa')
# plt.scatter(episodes_list, return_list3, s=0.5, label='Q-learning')

# # 画平滑曲线
# def smooth(data, window_size):
#     """对数据进行滑动平均平滑处理"""
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
# # 设置滑动窗口大小
# window_size = 8
# smoothed_return_list1 = smooth(return_list1, window_size)
# smoothed_return_list2 = smooth(return_list2, window_size)
# smoothed_return_list3 = smooth(return_list3, window_size)
# smoothed_episodes_list = list(range(len(smoothed_return_list1)))
# # 画平滑曲线
# plt.plot(smoothed_episodes_list, smoothed_return_list1, linewidth=0.99, label='Sarsa')
# plt.plot(smoothed_episodes_list, smoothed_return_list2, linewidth=0.99, label='5-step Sarsa')
# plt.plot(smoothed_episodes_list, smoothed_return_list3, linewidth=0.99, label='Q-learning')

# 画折线图
plt.plot(episodes_list, return_list1, linewidth=0.99, label='Sarsa')
plt.plot(episodes_list, return_list2, linewidth=0.99, label='5-step Sarsa')
plt.plot(episodes_list, return_list3, linewidth=0.99 ,label='Q-learning')
plt.plot(episodes_list, return_list4, linewidth=0.99 ,label='Q-learning-epsilon=0')

plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa, 5-step Sarsa, Q-learning on {}'.format('Cliff Walking'))
plt.legend()
plt.show()

print("\n",np.array(return_list1).mean(), np.array(return_list1).std())
print(np.array(return_list2).mean(), np.array(return_list2).std())
print(np.array(return_list3).mean(), np.array(return_list3).std())

# print(Qtable == agent.Q_table)

