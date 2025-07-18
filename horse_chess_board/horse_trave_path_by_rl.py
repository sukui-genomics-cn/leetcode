import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from knight_tour_env import KnightTourEnv



class DQNAgent:
    def __init__(self, board_size=8, device='cpu'):
        self.device = device
        self.board_size = board_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.model = self._build_model().to(device)
        self.target_model = self._build_model().to(device)
        self.update_target_model()
    
    def _build_model(self):
        """构建神经网络模型"""
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.board_size * self.board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 8)  # 8个可能的移动方向
        )
        return model
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到记忆回放"""
        self.memory.append((state, action, reward, next_state, done))
    
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
        
        minibatch = random.sample(self.memory, self.batch_size)
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
        
        optimizer = optim.Adam(self.model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    

def train_agent(episodes=1000, board_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KnightTourEnv(board_size, device=device)
    agent = DQNAgent(board_size, device=device)
    scores = []
    losses = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:  # 无有效移动
                break
                
            action = agent.act(state, valid_moves)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            loss = agent.replay()
            if loss:
                losses.append(loss)
        
        # 每10轮更新目标网络
        if e % 10 == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        
        if e % 50 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, scores, losses

# 训练智能体
agent, scores, losses = train_agent(episodes=5000, board_size=32)

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.title('Rewards per Episode')
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.show()