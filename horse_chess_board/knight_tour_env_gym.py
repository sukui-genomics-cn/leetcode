import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class KnightsTourEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 2}
    
    def __init__(self, board_size=8, render_mode: Optional[str] = "rgb_array"):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        
        # 定义动作空间 (8种可能的骑士移动)
        self.action_space = spaces.Discrete(8)
        
        # 定义观察空间 (4通道的棋盘表示)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(5, board_size, board_size),
            dtype=np.float32
        )
        
        # 骑士的8种可能移动 (国际象棋中马的走法)
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # 初始化状态变量
        self.board = None
        self.current_pos = None
        self.visited = None
        self.steps = None
        self.visited_order = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # 初始化棋盘状态
        self.board = np.zeros((self.board_size, self.board_size))
        self.current_pos = (
            self.np_random.integers(0, self.board_size),
            self.np_random.integers(0, self.board_size)
        )
        self.visited = [self.current_pos]
        self.visited_order = {self.current_pos: 1}
        self.steps = 0
        
        # 初始渲染
        if self.render_mode == 'human':
            self.render()
            
        return self._get_obs(), self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        """获取当前观察状态 (4通道表示)"""
        obs = np.zeros((5, self.board_size, self.board_size))
        
        # 通道0: 已访问位置
        for pos in self.visited:
            obs[0, pos[0], pos[1]] = 1
            
        # 通道1: 当前位置热图
        obs[1, self.current_pos[0], self.current_pos[1]] = 1
        
        # 通道2: 步数归一化
        obs[2, :, :] = self.steps / (self.board_size**2 * 2)
        
        # 通道3: 最近5步轨迹
        for i, pos in enumerate(self.visited[-5:]):
            obs[3, pos[0], pos[1]] = 0.8 - i * 0.15

        # 通道4: 下一步的合法移动
        valid_moves = self._get_valid_moves()
        for move in valid_moves:
            new_pos = (
                self.current_pos[0] + self.knight_moves[move][0],
                self.current_pos[1] + self.knight_moves[move][1]
            )
            obs[4, new_pos[0], new_pos[1]] = 1

            
        return obs.astype(np.float32)
    
    def _get_info(self) -> dict:
        """获取环境信息"""
        return {
            'current_position': self.current_pos,
            'visited_count': len(self.visited),
            'coverage': len(self.visited) / self.board_size**2,
            'valid_moves': self._get_valid_moves()
        }
    
    def _get_valid_moves(self) -> list:
        """获取当前合法动作索引"""
        valid_moves = []
        for i, move in enumerate(self.knight_moves):
            new_pos = (
                self.current_pos[0] + move[0],
                self.current_pos[1] + move[1]
            )
            if (0 <= new_pos[0] < self.board_size and 
                0 <= new_pos[1] < self.board_size and 
                new_pos not in self.visited):
                valid_moves.append(i)
        return valid_moves
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步动作"""
        move = self.knight_moves[action]
        new_pos = (
            self.current_pos[0] + move[0],
            self.current_pos[1] + move[1]
        )
        
        # 检查移动是否有效
        if not (0 <= new_pos[0] < self.board_size and 
                0 <= new_pos[1] < self.board_size):
            return self._get_obs(), -10, True, False, self._get_info()
        
        if new_pos in self.visited:
            return self._get_obs(), -5, True, False, self._get_info()
        
        # 更新状态
        self.current_pos = new_pos
        self.visited.append(new_pos)
        self.visited_order[new_pos] = len(self.visited)
        self.steps += 1
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查终止条件
        terminated = len(self.visited) == self.board_size**2
        truncated = False  # 可以设置最大步数限制
        
        # 渲染
        if self.render_mode == 'human':
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(self) -> float:
        """计算奖励值"""
        # 基础奖励
        reward = 1.0
        
        # 鼓励访问新区域
        coverage = len(self.visited) / self.board_size**2
        reward += coverage * 5.0

        # # 下一步的合法移动数量奖励
        valid_moves = self._get_valid_moves()
        if valid_moves:
            reward += len(valid_moves) * 0.5
        
        # 完成奖励
        if len(self.visited) == self.board_size**2:
            reward += 100
            
        # 无效移动惩罚 (提前在step中处理)
        
        return reward
    
    def render(self):
        """必须实现render方法"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self):
        """渲染当前棋盘状态"""
        if self.render_mode is None:
            return
            
        # 创建画布
        img = np.zeros((self.board_size, self.board_size, 3))
        
        # 绘制棋盘背景
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 0:
                    img[i, j] = [1.0, 0.9, 0.8]  # 浅色格子
                else:
                    img[i, j] = [0.5, 0.3, 0.1]  # 深色格子
        
        # 标记已访问的位置
        for pos in self.visited:
            img[pos[0], pos[1]] = [0.2, 0.6, 0.2]  # 绿色
            
        # 标记当前位置
        img[self.current_pos[0], self.current_pos[1]] = [0.8, 0.2, 0.2]  # 红色
        
        # 绘制移动序号
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        for pos, order in self.visited_order.items():
            ax.text(pos[1], pos[0], str(order), 
                   ha='center', va='center', 
                   color='white', fontsize=12)
        
        ax.set_title(f"Knight's Tour (Step {self.steps}, Coverage {len(self.visited)/self.board_size**2:.1%})")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if self.render_mode == 'human':
            plt.pause(0.5)  # 控制渲染速度
            plt.close()
        else:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.close()
            return img
    

    def close(self):
        """关闭环境"""
        plt.close('all')
        if hasattr(self, 'render_window'):
            import pygame
            pygame.display.quit()
            pygame.quit()

    # 添加wrapper属性访问支持
    def get_wrapper_attr(self, name):
        """支持wrapper属性访问"""
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def test_environment():
    # 创建环境
    env = KnightsTourEnv(board_size=5, render_mode='human')
    
    # 测试随机策略
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 随机选择合法动作
        valid_moves = info['valid_moves']
        if not valid_moves:
            print("No valid moves left!")
            break
            
        action = np.random.choice(valid_moves)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Total reward: {total_reward}")
    print(f"Coverage: {info['coverage']:.1%}")
    env.close()

if __name__ == "__main__":
    test_environment()