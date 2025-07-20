import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class KnightTourEnv:
    def __init__(self, board_size=8, device='cpu'):
        self.board_size = board_size
        self.device = device
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        # 随机起始位置
        self.current_pos = (np.random.randint(0, self.board_size), 
                           np.random.randint(0, self.board_size))
        self.board[self.current_pos] = 1
        self.visited =[self.current_pos]
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """改进后的状态表示"""
        state = np.zeros((4, self.board_size, self.board_size))
        # 通道0: 已访问位置
        for pos in self.visited:
            state[0, pos[0], pos[1]] = 1
        # 通道1: 当前位置热图
        state[1, self.current_pos[0], self.current_pos[1]] = 1
        # 通道2: 步数归一化
        state[2, :, :] = self.steps / (self.board_size**2 * 2)
        # 通道3: 最近5步轨迹
        for i, pos in enumerate(self.visited[-5:]):
            state[3, pos[0], pos[1]] = 0.8 - i*0.15
        return torch.FloatTensor(state).to(self.device)
    
    def get_valid_moves(self):
        """返回当前可用的移动动作索引"""
        valid_moves = []
        for i, move in enumerate(self.knight_moves):
            new_pos = (self.current_pos[0] + move[0], 
                       self.current_pos[1] + move[1])
            if (0 <= new_pos[0] < self.board_size and 
                0 <= new_pos[1] < self.board_size and 
                new_pos not in self.visited):
                valid_moves.append(i)
        return valid_moves
    
    def step(self, action):
        """执行一个动作，返回(next_state, reward, done)"""
        move = self.knight_moves[action]
        new_pos = (self.current_pos[0] + move[0], 
                   self.current_pos[1] + move[1])
        
        # 检查移动是否有效
        if not (0 <= new_pos[0] < self.board_size and 
                0 <= new_pos[1] < self.board_size):
            return self._get_state(), -10, True  # 非法移动，结束
        
        if new_pos in self.visited:
            return self._get_state(), -5, True  # 重复访问，结束
        
        # 更新状态
        self.current_pos = new_pos
        self.visited.append(new_pos)
        self.board[new_pos] = 1
        self.steps += 1
        

        # 基础奖励
        reward = 1.0
        
        # 鼓励访问新区域
        if self.current_pos not in self.visited:
            reward += 3.0
            
        # 动态调整系数
        coverage = len(self.visited) / self.board_size**2
        reward *= (1 + coverage**2)

        # 完成奖励
        if len(self.visited) == self.board_size**2:
            reward += 100
            done = True
        else:
            done = False
            
        # 无效移动惩罚
        if not self.get_valid_moves() and len(self.visited) < self.board_size**2:
            reward -= 10 * (self.board_size**2 - len(self.visited))
        
        return self._get_state(), reward, done
    
    @staticmethod
    def draw_knight_tour(path, board_size=8):
        """
        Visualize the knight's tour path on a chessboard.
        
        Args:
            path: List of tuples representing the knight's path (x, y coordinates)
            board_size: Size of the chessboard (default: 8x8)
        """
        # Initialize figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create chessboard pattern
        board = np.zeros((board_size, board_size))
        board[1::2, ::2] = 1  # Odd rows, even columns
        board[::2, 1::2] = 1  # Even rows, odd columns
        
        # Display chessboard
        ax.imshow(board, cmap='binary', interpolation='nearest')
        
        # Configure grid and labels
        ax.set_xticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, board_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xticklabels(np.arange(1, board_size+1))
        ax.set_yticklabels(np.arange(1, board_size+1))
        ax.tick_params(axis='both', which='both', length=0)
        
        # Extract coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        
        # Draw path lines
        for i in range(len(path)-1):
            x_start, y_start = path[i]
            x_end, y_end = path[i+1]
            ax.plot([x_start, x_end], [y_start, y_end], 'r-', linewidth=2)
        
        # Mark start and end points
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'bs', markersize=10, label='End')
        
        # Add knight symbol at end position
        ax.text(x_coords[-1], y_coords[-1], '♞', fontsize=20, 
                ha='center', va='center', color='blue')
        
        # Add move numbers
        for i, (x, y) in enumerate(path):
            ax.text(x, y, str(i+1), ha='center', va='center', 
                    color='red', fontsize=16, weight='bold')
        
        plt.title("Knight's Tour", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    env = KnightTourEnv()
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("No valid moves available. Ending tour.")
            break
        action = random.choice(valid_moves)  # 随机选择一个合法动作
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    
    print(f"Total reward: {total_reward}")
    env.draw_knight_tour(list(env.visited), board_size=env.board_size)