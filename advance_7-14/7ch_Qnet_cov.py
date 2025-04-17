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
        self.buffer = collections.deque(maxlen=capacity)
        # 队列,先进先出
    def add(self, frame:np.ndarray, action, reward, next_frame:np.ndarray, done):  # 将数据加入buffer
        self.buffer.append((frame, action, reward, next_frame, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        frame, action, reward, next_state, done = zip(*transitions)
        # print(f"frame type: {type(frame)}, action type: {type(action)}, reward type: {type(reward)}, next_state type: {type(next_state)}, done type: {type(done)}")
        return (frame), action, reward, np.array(next_state), done
        # return (
        #     torch.tensor(frame, dtype=torch.float).to(self.device),
        #     torch.tensor(action, dtype=torch.long).to(self.device),
        #     torch.tensor(reward, dtype=torch.float).to(self.device),
        #     torch.tensor(np.array(next_state), dtype=torch.float).to(self.device),
        #     torch.tensor(done, dtype=torch.float).to(self.device),
        # )

class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, device, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=8, stride=4)
        self.residual_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=1, stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=1, stride=1)
            ) for _ in range(50)
        ])
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

        # 这里batah_size = 1
        self.x_in = torch.zeros(1, in_channels, 84, 84).to(device)  # 输入的图片大小

    def _forward(self, x:torch.Tensor):
        # x应该是一个4维的张量，（ in_channels, frame_channels, height, width）
        # 每张图片的大小是84*84，通道数是4
        # 计算过程：(84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7

        x = F.relu(self.conv1(x))
        for block in self.residual_blocks:
            x = block(x) + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)  # 从第 1 维开始展平
        x = F.relu(self.fc4(x))
        return self.head(x)
    
    def forward(self, x:np.ndarray):
        if isinstance(x, tuple):
            x = np.array(x)  # 将 tuple 转换为 np.ndarray
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, but got {type(x)}")
        # print(f"x type: {type(x)}, x shape: {x.shape}")
        assert x.shape == (400, 600, 3), f"Expected shape (400, 600, 3), but got {x.shape}"
        # x的shape应该是(400,600,3)
        x = torch.from_numpy(x).float().to(self.x_in.device)
        # 转化为灰度图
        x = torch.mean(x, dim=2)  # 将RGB转化为灰度图
        # 现在x的shape是(400,600)
        # 进行resize
        x = torch.nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False)
        # 现在x的shape是(1, 1, 84, 84)
        # 像素值归一化到[0, 1]
        x = x / 255.0
        # 将self.x_in的最旧的数据剔除
        self.x_in = torch.roll(self.x_in, shifts=-1, dims=1)
        # 然后将x填入 self.x_in 的最新的数据，self.x_in的shape是(1, 4, 84, 84)
        self.x_in[-1] = x

        x = self._forward(self.x_in)
        return x

    
class DQN_cov:
    def __init__(self, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device):
        self.action_dim = action_dim
        self.device = device
        # 当前网络，omega
        self.q_net = ConvolutionalQnet(self.action_dim, device).to(device)
        # 目标网络，omega-
        self.target_q_net = ConvolutionalQnet(self.action_dim, device).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.count = 0
    
    def select_action(self, frame, training=True):
        # epsilon-greedy策略
        # 更新epsilon
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                # state = torch.tensor(state, dtype=torch.float).to(self.device)
                q_values = self.q_net(frame)
                action = torch.argmax(q_values).item()
                return action
            
    def update(self, batch_size, replay_buffer:ReplayBuffer):
        
        # 从buffer中采样数据,数量为batch_size
        frames, actions, rewards, next_frames, dones = replay_buffer.sample(batch_size)
        for i in range(len(frames)):

            # 计算当前网络的Q值
            frame = frames[i]
            # action = actions[i]
            reward = rewards[i]
            next_frame = next_frames[i]
            done = dones[i]
            action = torch.tensor(actions[i], dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(1)
            # rewards = torch.tensor(rewards[i], dtype=torch.float).to(self.device)
            # next_frame = torch.tensor(next_frames[i], dtype=torch.float).to(self.device)
            # done = torch.tensor(dones[i], dtype=torch.float).to(self.device)
            # print(f"frame type: {(frame.shape)}, action type: {type(action)}, reward type: {type(reward)}, next_state type: {type(next_state)}, done type: {type(done)}")
            tmp1 = self.q_net(frame)
            # print(f"tmp1 shape: {tmp1.shape}, action shape: {action.shape}")
            q_values = tmp1.gather(1, action)
            # 计算目标网络的Q值
            next_q_values = self.target_q_net(next_frame).max(1)[0].detach()
            target_q_values = reward + (1 - done) * self.gamma * next_q_values
            target_q_values = target_q_values.unsqueeze(1) 
            # 计算损失函数
            loss = F.mse_loss(q_values, target_q_values)
            # 更新当前网络的参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 更新目标网络
            if self.count % self.target_update == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.count += 1


lr = 1e-4
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.75
epsilon_decay = 0.001
target_update = 20
buffer_size = 10000
minimal_size = 1000
batch_size = 64
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
agent = DQN_cov(action_dim, lr, gamma, epsilon, epsilon_decay,
                 target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=num_episodes // 10, desc='Iteration %d' % i) as pbar:
        for episode in range(num_episodes // 10):
            state, info = env.reset()
            done = False
            total_reward = 0
            frame = env.render()
            if (episode + 1) % 50 == 0:
                frames = []
                save_path = 'vid'
            while not done:
                action = agent.select_action(frame)
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                next_frame = env.render()
                # plt.imshow(next_frame)
                # onesnp = 255*np.ones((400, 600, 3), dtype=np.uint8)
                # 看看onesnp和next_frame是否全部相等
                # print(np.array_equal(onesnp, next_frame))
                
                replay_buffer.add(frame, action, reward, next_frame, done)
                # frame = env.render()
                if len(replay_buffer.buffer) > batch_size:
                    agent.update(batch_size, replay_buffer)
                if (episode + 1) % 50 == 0:
                    frames.append(frame)
                state = next_state
                frame = next_frame
            return_list.append(total_reward)
            pbar.set_postfix(episode=episode, reward=total_reward)
            if (episode + 1) % 50 == 0:
                frames.append(env.render())
                # print(frames[-1])
                imageio.mimsave(os.path.join(save_path, f'episode_{i}_{episode}.gif'), frames, fps=30)
            pbar.update(1)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episodes')
plt.savefig(
    'pic/{}_Cov_{}.png'.format(env_name, time.strftime('%Y-%m-%d_%H-%M-%S')))
plt.show()