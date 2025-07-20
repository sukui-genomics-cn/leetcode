import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List


class KnightsTourEnv(gym.Env):
    """Custom environment for solving the Knight's Tour problem.
    
    The Knight's Tour is a sequence of moves by a knight on a chessboard
    such that the knight visits every square exactly once.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 2}
    
    def __init__(self, board_size: int = 8, render_mode: Optional[str] = "rgb_array"):
        """Initialize the Knight's Tour environment.
        
        Args:
            board_size: Size of the chessboard (default 8x8)
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        
        # Action space: 8 possible knight moves
        self.action_space = spaces.Discrete(8)
        
        # Observation space: 5-channel board representation
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(5, board_size, board_size),
            dtype=np.float32
        )
        
        # Knight's possible moves (dx, dy)
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # Initialize state variables
        self.board = None
        self.current_pos = None
        self.visited = None
        self.steps = None
        self.visited_order = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state.
        
        Returns:
            observation: Initial board observation
            info: Additional environment information
        """
        super().reset(seed=seed)
        
        # Initialize board state
        self.board = np.zeros((self.board_size, self.board_size))
        self.current_pos = (
            self.np_random.integers(0, self.board_size),
            self.np_random.integers(0, self.board_size)
        )
        self.visited = [self.current_pos]
        self.visited_order = {self.current_pos: 1}
        self.steps = 0
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_obs(), self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation as multi-channel array.
        
        Returns:
            ndarray: 5-channel observation array
                [0] Visited positions
                [1] Current position
                [2] Normalized step count
                [3] Recent move trajectory
                [4] Valid next moves
        """
        obs = np.zeros((5, self.board_size, self.board_size))
        
        # Channel 0: Visited positions
        for pos in self.visited:
            obs[0, pos[0], pos[1]] = 1
            
        # Channel 1: Current position
        obs[1, self.current_pos[0], self.current_pos[1]] = 1
        
        # Channel 2: Normalized step count
        obs[2, :, :] = self.steps / (self.board_size**2 * 2)
        
        # Channel 3: Last 5 moves
        for i, pos in enumerate(self.visited[-5:]):
            obs[3, pos[0], pos[1]] = 0.8 - i * 0.15

        # Channel 4: Valid next moves
        valid_moves = self._get_valid_moves()
        for move in valid_moves:
            new_pos = (
                self.current_pos[0] + self.knight_moves[move][0],
                self.current_pos[1] + self.knight_moves[move][1]
            )
            obs[4, new_pos[0], new_pos[1]] = 1
            
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional environment information.
        
        Returns:
            dict: Contains current position, coverage, and valid moves
        """
        return {
            'current_position': self.current_pos,
            'visited_count': len(self.visited),
            'coverage': len(self.visited) / self.board_size**2,
            'valid_moves': self._get_valid_moves()
        }
    
    def _get_valid_moves(self) -> List[int]:
        """Get indices of currently valid moves.
        
        Returns:
            list: Indices of valid moves from current position
        """
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
        """Execute one environment step.
        
        Args:
            action: Move index to execute
            
        Returns:
            observation: New board state
            reward: Step reward
            terminated: Episode termination flag
            truncated: Early truncation flag
            info: Additional environment info
        """
        move = self.knight_moves[action]
        new_pos = (
            self.current_pos[0] + move[0],
            self.current_pos[1] + move[1]
        )
        
        # Check for invalid move
        if not (0 <= new_pos[0] < self.board_size and 
                0 <= new_pos[1] < self.board_size):
            return self._get_obs(), -10, True, False, self._get_info()
        
        if new_pos in self.visited:
            return self._get_obs(), -5, True, False, self._get_info()
        
        # Update state
        self.current_pos = new_pos
        self.visited.append(new_pos)
        self.visited_order[new_pos] = len(self.visited)
        self.steps += 1
        
        reward = self._calculate_reward()
        terminated = len(self.visited) == self.board_size**2
        truncated = False
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(self) -> float:
        """Calculate step reward.
        
        Returns:
            float: Reward value combining:
                - Base move reward
                - Coverage bonus
                - Valid move options bonus
                - Completion bonus
        """
        reward = 1.0
        coverage = len(self.visited) / self.board_size**2
        reward += coverage * 5.0

        valid_moves = self._get_valid_moves()
        if valid_moves:
            reward += len(valid_moves) * 0.5
        
        if len(self.visited) == self.board_size**2:
            reward += 100
            
        return reward
    
    def render(self):
        """Render current board state."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self):
        """Generate visualization frame."""
        if self.render_mode is None:
            return
            
        # Create board image
        img = np.zeros((self.board_size, self.board_size, 3))
        
        # Draw checkerboard pattern
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 0:
                    img[i, j] = [1.0, 0.9, 0.8]  # Light squares
                else:
                    img[i, j] = [0.5, 0.3, 0.1]  # Dark squares
        
        # Mark visited positions
        for pos in self.visited:
            img[pos[0], pos[1]] = [0.2, 0.6, 0.2]  # Green
            
        # Mark current position
        img[self.current_pos[0], self.current_pos[1]] = [0.8, 0.2, 0.2]  # Red
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        # Add move numbers
        for pos, order in self.visited_order.items():
            ax.text(pos[1], pos[0], str(order), 
                   ha='center', va='center', 
                   color='white', fontsize=12)
        
        ax.set_title(f"Knight's Tour (Step {self.steps}, Coverage {len(self.visited)/self.board_size**2:.1%})")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if self.render_mode == 'human':
            plt.pause(0.5)
            plt.close()
        else:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.close()
            return img
    
    def close(self):
        """Clean up environment resources."""
        plt.close('all')


def test_environment():
    """Test the Knight's Tour environment with random actions."""
    env = KnightsTourEnv(board_size=5, render_mode='human')
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
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