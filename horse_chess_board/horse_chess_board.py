"""
思路: 深度搜索算法
1. 创建2维棋盘
2. 将当前位置设置为已访问, 且每走一步, 计数器+1
3. 根据当前位置计算可走的位置
4. 遍历获取可以走的位置列表, 如果该位置为走过, 则访问(递归, 从第2步开始), 否则放弃位置
5. 判断马是否完成任务.
"""
from typing import Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class HourseTraveling:
    def __init__(self, rows:int, cols:int):
        self.rows:int = rows
        self.cols:int = cols
        self.init_pos = None
        self.chess_board = np.zeros(shape=[row, col], dtype=np.int16)
        self.steps = []
        self.counts:int = 0
        self.last_step = None

    def get_ava_pos(self, row, col):
        """
        get available positions for the horse from the current position (row, col).
        :param row: current row
        :param col: current col
        :return: list of available positions as list [pos1, pos2, ...], where pos is a tuple (row, col)
        """
        moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        ava_pos = []
        
        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols and not self.chess_board[new_row][new_col]:
                if (new_row, new_col) == self.last_step and len(self.steps) != self.rows * self.cols - 1:
                    # only allow returning to the initial position if all other positions have been visited
                    pass
                else:
                    ava_pos.append((new_row, new_col))

        return ava_pos
    
    def set_init_pos(self, row:int, col:int, last_row=None, last_col=None):
        """
        set the initial position of the horse on the chess board.
        :param row: initial row
        :param col: initial col
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.counts = 0
            self.init_pos = (row, col)
            if last_row is not None and last_col is not None:
                self.last_step = (last_row, last_col)
            self.chess_board = np.zeros(shape=[self.rows, self.cols], dtype=np.int16)
            self.steps = []
            logger.info(f"initial position set to ({row}, {col})")
        else:
            raise ValueError(f"Initial position is out of bounds. borad row: {self.rows}, col: {self.cols}. Given: ({row}, {col})")
        

    def traving_start(self, start_pos=Optional[Tuple], end_pos=None):
        """
        start the horse traveling path from a given start position to an end position.
        :param start_pos: tuple (row, col) of the starting position
        :param end_pos: tuple (row, col) of the ending position
        :return: step_path: list of tuples representing the path taken by the horse
        """
        if start_pos is not None:
            self.set_init_pos(start_pos[0], start_pos[1])
        if end_pos is not None:
            self.last_step = end_pos

        if self.init_pos is None:
            raise ValueError("Initial position must be set before starting the tour.")
        
        logger.info(f"Starting position: {self.init_pos}, Last step: {self.last_step}")
        return self.next_step(self.init_pos[0], self.init_pos[1])

    def next_step(self, row:int, col:int):
        """
        step to the next position (row, col) on the chess board, marking it as visited.
        :param row: next row to step
        :param col: next col to step
        :return: True if the horse has completed the chess board, False starting to recurisive next step
        """
        if self.counts == 0:
            # self.set_init_pos(row, col)
            self.counts += 1
        else:
            self.chess_board[row][col] = len(self.steps) + 1
            self.steps.append((row, col))
            self.counts += 1

        if self.is_complete(row, col):
            logger.info(f"last position: ({row}, {col}), all steps: {self.counts}")
            return (row, col)
        else:
            
            ava_pos = self.get_ava_pos(row, col)
            ava_pos = sorted(ava_pos, key=lambda pos: len(self.get_ava_pos(pos[0], pos[1])), reverse=False)

            for next_row, next_col in ava_pos:
                logger.info(f'from position: ({row}, {col}), to next position: ({next_row}, {next_col})')
                ava_pos = self.next_step(next_row, next_col)
                if ava_pos:
                    return ava_pos[0], ava_pos[1]
                else:
                    logger.info(f'recursive backtracking: ({row}, {col})')
                    self.chess_board[next_row][next_col] = False
                    self.steps.pop()


    def is_complete(self,row, col):
        """
        check if the horse has completed the chess board.
        :param row: current row
        :param col: current col
        :return: True if the horse has visited all squares, False otherwise
        """
        if len(self.steps) == self.rows * self.cols and (row, col) == self.last_step:                                                                 
            logger.info(f'horse chess board is complete. steps: \n{self.steps}')
            return True
        return False

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
                    color='red', fontsize=8, weight='bold')
        
        plt.title("Knight's Tour", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

def test_horseTravelingPath():
    """
    Test the horse traveling path algorithm with a specific starting position.
    """
    row = 8
    col = 8
    chess_board = HourseTraveling(row, col)
    # chess_board.set_init_pos(1, 2)
    chess_board.traving_start(1, 2)
    assert chess_board.is_complete(1, 2) == False, "The knight's tour should not be complete yet."
    
    # Check if the knight can return to the initial position after visiting all other squares
    chess_board.steps.append((1, 2))
    assert chess_board.is_complete(1, 2) == True, "The knight's tour should be complete now."

if __name__ == '__main__':
    row = 8
    col = 8
    chess_board = HourseTraveling(row, col)
    start_time = time.time()
    chess_board.traving_start(start_pos=(1, 2), end_pos=(1, 2))
    if chess_board.steps:
        # Draw the knight's tour path
        chess_board.draw_knight_tour(chess_board.steps, board_size=row)
    else:
        logger.info("No valid knight's tour path found.")
    end_time = time.time()
    logger.info(f'cost time: {end_time - start_time}')
