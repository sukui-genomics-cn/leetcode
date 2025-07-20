import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from knight_tour_env import KnightTourEnv


class EnhancedDQN(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*(board_size//2)**2, 256),
            nn.LayerNorm(256),  # 添加归一化
            nn.ReLU(),
            nn.Linear(256, 8)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNAgent:
    def __init__(self, board_size=8, batch_size:int=64, load_path:str=None, device='cpu'):
        self.device = device
        self.board_size = board_size
        self.priority_memory = deque(maxlen=10000)  # 存储(td_error, experience)
        self.beta = 0.4  # 优先度采样参数

        if load_path:
            # 如果提供了加载路径，则从文件加载
            self._load_initial_state(load_path)
            
        else:
            self.memory = deque(maxlen=10000)
            self.gamma = 0.95  # 折扣因子
            self.epsilon = 1.0  # 探索率
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.batch_size = batch_size
            self.model = self._build_model().to(device)
            self.target_model = self._build_model().to(device)
            self.optimizer = optim.Adam(self.model.parameters())
            self.update_target_model()

    def _load_initial_state(self, load_path):
        """从检查点初始化状态"""
        assert os.path.exists(load_path), f"Checkpoint file {load_path} does not exist."
        checkpoint = torch.load(load_path, map_location=self.device)
        self.memory = deque(checkpoint['memory'], maxlen=10000)
        self.gamma = checkpoint['gamma']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_min = checkpoint['epsilon_min']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.batch_size = checkpoint['batch_size']
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _build_model(self):
        """构建神经网络模型"""
        model = EnhancedDQN(self.board_size)
        model.to(self.device)
        return model
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        # 计算初始TD误差
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(self.device)
            q_val = self.model(state_t)[0][action]
            next_q = self.target_model(next_state.unsqueeze(0).to(self.device)).max()
            td_error = abs(reward + self.gamma * next_q * (1-done) - q_val.item())
        
        self.memory.append((state, action, reward, next_state, done))
        self.priority_memory.append((td_error + 1e-5, len(self.memory)-1))  # 防止0优先级
    
    def act(self, state, valid_moves):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)  # 随机探索
        
        state = state.unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            act_values = self.model(state)
        
        # 只考虑有效动作
        act_values = act_values.squeeze().cpu().numpy()
        valid_act_values = [act_values[i] for i in valid_moves]
        return valid_moves[np.argmax(valid_act_values)]
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 按优先级采样
        probs = np.array([x[0].cpu() for x in self.priority_memory])
        probs /= probs.sum()
        indices = np.random.choice(len(self.priority_memory), 
                                 size=self.batch_size,
                                 p=probs)
        indices = indices.tolist()  # 转换为Python列表
        minibatch = [self.memory[i] for i in indices]  # 列表推导式索引
        states = torch.stack([x[0] for x in minibatch]).to(self.device)
        actions = torch.tensor([x[1] for x in minibatch]).to(self.device)
        rewards = torch.tensor([x[2] for x in minibatch], 
                             dtype=torch.float32).to(self.device)
        next_states = torch.stack([x[3] for x in minibatch]).to(self.device)
        dones = torch.tensor([x[4] for x in minibatch], 
                            dtype=torch.float32).to(self.device)
        
        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新
        criterion = nn.MSELoss()
        loss = criterion(current_q.squeeze(), target_q)
        
        # optimizer = optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

def save_agent(agent:DQNAgent, path='knight_agent.pth'):
    """保存Agent的完整状态"""
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'memory': list(agent.memory),
        'epsilon': agent.epsilon,
        'board_size': agent.board_size,
        'gamma': agent.gamma,
        'epsilon_min': agent.epsilon_min,
        'epsilon_decay': agent.epsilon_decay,
        'batch_size': agent.batch_size
    }
    torch.save(checkpoint, path)
    print(f"Agent saved to {path}") 

def train_agent(episodes=1000, board_size=8, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KnightTourEnv(board_size, device=device)
    agent = DQNAgent(board_size, batch_size=batch_size, device=device)
    scores = []
    losses = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        invalid_move_count = 0  # 跟踪无效移动
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            # 无有效移动时的强化惩罚处理
            if not valid_moves:
                # 给予惩罚并提前结束本轮
                reward = -10 * (env.board_size**2 - len(env.visited))  # 剩余未访问格子的惩罚
                agent.remember(state, 0, reward, state, True)  # 存入记忆
                losses.append(agent.replay())  # 强制学习这次惩罚
                invalid_move_count += 1
                break
                
            action = agent.act(state, valid_moves)
            next_state, reward, done = env.step(action)
            
            # 动态奖励调整
            remaining = env.board_size**2 - len(env.visited)
            reward += 0.5 * (1 - remaining/(env.board_size**2))  # 越后期奖励越高
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # 优先经验回放（Prioritized Experience Replay）
            loss = agent.replay()
            if loss:
                losses.append(loss)
        
        # 回合结束后的处理
        scores.append(total_reward)
        
        # 动态ε衰减（后期降低探索）
        if e > episodes//2:
            agent.epsilon = max(agent.epsilon_min, 
                            agent.epsilon * agent.epsilon_decay**2)
        else:
            agent.epsilon = max(agent.epsilon_min, 
                            agent.epsilon * agent.epsilon_decay)
        
        # 每10轮更新目标网络
        if e % 10 == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        
        # 每100轮打印进度
        if e % 100 == 0:
            coverage = len(env.visited)/env.board_size**2
            print(f"Ep {e}: Reward {total_reward:.1f}, "
                f"Coverage {coverage:.1%}, "
                f"Invalid {invalid_move_count}, "
                f"ε {agent.epsilon:.3f}")
    
    return agent, scores, losses

# 训练智能体
agent, scores, losses = train_agent(episodes=5000, board_size=5, batch_size=256)
save_agent(agent, 'knight_agent_final.pth')

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.title('Rewards per Episode')
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.show()


def test_agent(agent:DQNAgent, board_size=8):
    env = KnightTourEnv(board_size)
    state = env.reset()
    path = [env.current_pos]
    done = False
    
    while not done:
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("No valid moves left!")
            break
            
        action = agent.act(state, valid_moves)
        state, _, done = env.step(action)
        path.append(env.current_pos)
    
    # 可视化路径
    board = np.zeros((board_size, board_size))
    for i, pos in enumerate(path):
        board[pos] = i + 1
    
    plt.figure(figsize=(8, 8))
    plt.imshow(board, cmap='viridis')
    plt.colorbar(label='Move Number')
    for i in range(board_size):
        for j in range(board_size):
            if (i, j) in path:
                plt.text(j, i, int(board[i, j]), ha='center', va='center', color='white')
    plt.title('Knight\'s Tour Path')
    plt.show()
    
    print(f"Total moves: {len(path)}")
    print(f"Unique squares visited: {len(set(path))}")
    return path

# 测试训练好的智能体
agent = DQNAgent(board_size=5, load_path='knight_agent_final.pth')
path = test_agent(agent)