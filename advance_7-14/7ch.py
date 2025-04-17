import random
import gym
import imageio, os
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils as rl_utils
import time

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity, device):
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        # return np.array(state), action, reward, np.array(next_state), done
        return (
            torch.tensor(np.array(state), dtype=torch.float).to(self.device),
            torch.tensor(action, dtype=torch.long).to(self.device),
            torch.tensor(reward, dtype=torch.float).to(self.device),
            torch.tensor(np.array(next_state), dtype=torch.float).to(self.device),
            torch.tensor(done, dtype=torch.float).to(self.device),
        )

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    ''' 多层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.qnet1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim*4),
            torch.nn.ReLU(),
        )
        # 动态创建多个 qnet2 模块
        self.residual_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 4, hidden_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim * 4, hidden_dim * 4)
            ) for _ in range(1)
        ])
        self.qnet3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*4, hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim)
        )
    def forward(self, x):
        x = self.qnet1(x)
        # 残差网络
        for block in self.residual_blocks:
            x = block(x) + x  # 每个残差块的参数不同
 
        x = self.qnet3(x)
        return x
    

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon,epsilon_decay, target_update, device):
        self.action_dim = action_dim
        # 当前网络，omega
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络，omega-
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)  # 衰减epsilon
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = transition_dict['states']  # 已经是 tensor，无需再次转换
        actions = transition_dict['actions'].view(-1, 1)  # 调整形状即可
        rewards = transition_dict['rewards'].view(-1, 1)
        next_states = transition_dict['next_states']
        dones = transition_dict['dones'].view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


lr = 1e-4
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.75
epsilon_decay = 0.001
target_update = 20
buffer_size = 100000
minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='rgb_array')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size, device)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"env obs shape:{env.observation_space.shape}, State dimension: {state_dim}, Action dimension: {action_dim}")
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, epsilon_decay,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()
            if (i_episode + 1) % 50 == 0:
                frames = []
                save_path = 'vid'
                frames.append(env.render())
                # print(frames[-1])
            done = False
            while not done:
                if (i_episode + 1) % 50 == 0:
                    frames.append(env.render())
                # print("State:", state, "Shape:", np.shape(state))
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 50 == 0:
                imageio.mimsave(os.path.join(save_path, 'cartpole_{}_{}_{}.gif'.format(env_name, i_episode, time.strftime('%Y-%m-%d_%H-%M-%S')) ), frames)
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list, label='DQN')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
# plt.show()

# 计算滑动平均
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return, label='Moving Average')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
# 将图片保存到pic文件夹下，命名为日期+时间：
plt.savefig(
    'pic/{}_{}.png'.format(env_name, time.strftime('%Y-%m-%d_%H-%M-%S')))
# plt.show()
