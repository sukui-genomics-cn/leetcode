# Leetcode Project: Algorithmic Challenges and Solutions

This repository contains implementations of various algorithmic challenges, including prime factorization, the Knight's Tour problem, and custom environments for reinforcement learning. The solutions are implemented in Python and are designed to be modular and reusable.

## Features
- **Prime Factorization**: Efficient algorithms for generating primes, finding factors, and solving related mathematical problems.
- **Knight's Tour Problem**: A depth-first search solution for the classic chessboard traversal problem, with visualization support.
- **Reinforcement Learning Environment**: A custom Gym environment for solving the Knight's Tour problem using RL techniques.

## Usage

### Prime Factorization
Navigate to the `find_factor` directory and run the script:
```bash
python generate_primes.py
```
This script includes methods for generating primes, finding factors, and solving mathematical problems.

### Knight's Tour Problem
Navigate to the `horse_trave` directory and run the script:
```bash
python horse_trave.py --size 8 --start_row 0 --start_col 0 --end_row 7 --end_col 7
```
This will solve the Knight's Tour problem for an 8x8 chessboard starting at (0, 0) and ending at (7, 7). The solution path will be visualized.

### Reinforcement Learning Environment
Navigate to the `horse_trave` directory and run the environment test:
```bash
python ppo_knight_tour_custom_net.py
```
This will test the custom Gym environment for the Knight's Tour problem with random actions.

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Gymnasium

Install dependencies using:
```bash
git clone -b release https://github.com/sukui-genomics-cn/leetcode.git
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.
