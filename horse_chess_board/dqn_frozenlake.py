# Deep Reinforcement Learning: DQN on Custom FrozenLake Environments

# =============================================================================
# Imports and Setup
# =============================================================================

# Import necessary libraries
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Creating the Custom FrozenLake Environments
# =============================================================================

def create_random_map(size=10, holes=20, start=(2, 9), goal=(8, 8)):
    """
    Creates a random FrozenLake map with specified size and number of holes.
    """
    valid = False
    while not valid:
        # Initialize the map with all frozen tiles
        map = [['F'] * size for _ in range(size)]

        # Place the start and goal positions
        map[start[0]][start[1]] = 'S'
        map[goal[0]][goal[1]] = 'G'

        # Randomly place holes
        hole_count = 0
        while hole_count < holes:
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            if (i, j) != start and (i, j) != goal and map[i][j] == 'F':
                map[i][j] = 'H'
                hole_count += 1

        # Convert the map to the required format
        map_str = [''.join(row) for row in map]

        # Check if the map is solvable
        valid = is_map_solvable(map_str, start, goal)

    return map_str

def is_map_solvable(map_desc, start, goal):
    """
    Checks if the generated FrozenLake map is solvable.
    """
    size = len(map_desc)
    visited = set()
    stack = []

    start_idx = start[0] * size + start[1]
    goal_idx = goal[0] * size + goal[1]

    stack.append(start_idx)
    visited.add(start_idx)

    while stack:
        current = stack.pop()
        if current == goal_idx:
            return True
        row, col = divmod(current, size)
        # Explore neighboring cells
        for action in range(4):  # Up, Down, Left, Right
            new_row, new_col = row, col
            if action == 0 and row > 0:  # Up
                new_row -= 1
            elif action == 1 and row < size - 1:  # Down
                new_row += 1
            elif action == 2 and col > 0:  # Left
                new_col -= 1
            elif action == 3 and col < size - 1:  # Right
                new_col += 1
            else:
                continue
            if map_desc[new_row][new_col] != 'H':
                new_idx = new_row * size + new_col
                if new_idx not in visited:
                    visited.add(new_idx)
                    stack.append(new_idx)
    return False

# Define start and goal positions based on Panther ID: 002802988
start_pos = (2, 9)  # First two digits: (2, 9)
goal_pos = (8, 8)   # Last two digits: (8, 8)

# Create maps for both cases
map_case1 = create_random_map(size=10, holes=20, start=start_pos, goal=goal_pos)
map_case2 = create_random_map(size=10, holes=40, start=start_pos, goal=goal_pos)

# =============================================================================
# Map Visualization
# =============================================================================

def visualize_map(map_desc, title="Map"):
    size = len(map_desc)
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if map_desc[i][j] == 'S':
                grid[i, j] = 0.5  # Start position
            elif map_desc[i][j] == 'G':
                grid[i, j] = 0.75  # Goal position
            elif map_desc[i][j] == 'F':
                grid[i, j] = 1  # Frozen lake
            elif map_desc[i][j] == 'H':
                grid[i, j] = 0  # Hole
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='cool', origin='upper')
    plt.title(title)
    plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1],
                 format=plt.FuncFormatter(lambda val, loc: {0: 'H', 0.25: '', 0.5: 'S', 0.75: 'G', 1: 'F'}[val]))
    plt.show()

# Visualize the maps
print("Visualizing Map for Case 1:")
visualize_map(map_case1, title="Map for Case 1")
print("Visualizing Map for Case 2:")
visualize_map(map_case2, title="Map for Case 2")

# Initialize environments with deterministic dynamics (is_slippery=False)
env_case1 = gym.make("FrozenLake-v1", desc=map_case1, is_slippery=False)
env_case2 = gym.make("FrozenLake-v1", desc=map_case2, is_slippery=False)

# =============================================================================
# Defining the Deep Q-Network (DQN)
# =============================================================================

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        Initializes the neural network.
        """
        super(DQN, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Hidden layer to output layer
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# =============================================================================
# Implementing Experience Replay
# =============================================================================

class ReplayMemory:
    def __init__(self, capacity=50000):
        """
        Initializes the replay memory.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# =============================================================================
# Helper Functions
# =============================================================================

def one_hot_encode(state, state_size):
    """
    One-hot encodes the state.
    """
    if isinstance(state, tuple):
        state = state[0]
    elif isinstance(state, np.ndarray):
        state = state.item()
    state_encoded = np.zeros(state_size)
    state_encoded[state] = 1
    return state_encoded

def learn(experiences):
    """
    Updates the policy network based on a batch of experiences.
    """
    # Unpack experiences
    states, actions, rewards, next_states, dones = zip(*experiences)

    # Convert to numpy arrays for efficient conversion
    states = np.vstack(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.vstack(next_states)
    dones = np.array(dones).astype(int)

    # Convert to tensors and move to device
    states = torch.from_numpy(states).float().to(device)
    actions = torch.from_numpy(actions).unsqueeze(1).long().to(device)
    rewards = torch.from_numpy(rewards).unsqueeze(1).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.from_numpy(dones).unsqueeze(1).float().to(device)

    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q-values using the target network
    with torch.no_grad():
        q_next = target_net(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + (gamma * q_next * (1 - dones))

    # Compute loss
    loss = criterion(q_values, q_target)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# =============================================================================
# Adjusting the Reward Structure
# =============================================================================

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # Modify the reward
        if reward == 0 and not done:
            reward = -0.05  # Increased penalty for each step
        elif reward == 1:
            reward = 5  # Increased reward for reaching the goal
        else:
            reward = -5  # Increased penalty for falling into a hole
        return next_state, reward, done, info

# Wrap the environment with the updated RewardWrapper
env_case1 = gym.make("FrozenLake-v1", desc=map_case1, is_slippery=False)
env_case1 = RewardWrapper(env_case1)

# Similarly wrap the environment for Case 2
env_case2 = gym.make("FrozenLake-v1", desc=map_case2, is_slippery=False)
env_case2 = RewardWrapper(env_case2)

# =============================================================================
# Training the Agent for Case 1
# =============================================================================

# Adjusted Hyperparameters for Case 1
state_size = env_case1.observation_space.n  # Number of states
action_size = env_case1.action_space.n      # Number of actions
hidden_size = 256                           # Hidden layer size
batch_size = 64                             # Batch size for training
gamma = 0.95                                # Increased discount factor
epsilon = 1.0                               # Initial exploration rate
epsilon_min = 0.01                          # Minimum exploration rate
epsilon_decay = 0.995                       # Exploration decay rate
learning_rate = 0.005                       # Learning rate
target_update = 10                          # Frequency to update target network
num_episodes = 5000                         # Number of training episodes

# Initialize the policy network
policy_net = DQN(state_size, action_size, hidden_size).to(device)

# Initialize the target network
target_net = DQN(state_size, action_size, hidden_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Initialize the optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Initialize the replay memory
memory = ReplayMemory(50000)

# Define the loss function
criterion = nn.MSELoss()

def train(env, num_episodes):
    """
    Trains the DQN agent on the given environment.
    """
    global epsilon
    rewards = []
    successes = []
    losses = []
    success_count = 0  # Track number of successes

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        elif isinstance(state, np.ndarray):
            state = state.item()
        state = one_hot_encode(state, state_size)
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            elif isinstance(next_state, np.ndarray):
                next_state = next_state.item()
            next_state_encoded = one_hot_encode(next_state, state_size)
            total_reward += reward

            # Store experience
            memory.push(state, action, reward, next_state_encoded, done)

            # Move to next state
            state = next_state_encoded

            # Learning step
            if len(memory) > batch_size:
                experiences = memory.sample(batch_size)
                loss = learn(experiences)
                losses.append(loss)

        if reward == 5:  # Adjusted reward for reaching the goal
            success_count += 1

        rewards.append(total_reward)
        successes.append(success_count)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update the target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Successes: {success_count}, "
                  f"Epsilon: {epsilon:.4f}")
            success_count = 0  # Reset success count every 100 episodes

    return rewards, losses

# Reset epsilon
epsilon = 1.0

# Train the agent on Case 1
print("Training on Case 1 with adjusted hyperparameters and reward structure...")
rewards_case1, losses_case1 = train(env_case1, num_episodes)

# Plot the total rewards over episodes
plt.figure(figsize=(12, 6))
plt.plot(rewards_case1)
plt.title('Rewards over Episodes (Case 1)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Plot the loss over time
plt.figure(figsize=(12, 6))
plt.plot(losses_case1)
plt.title('Loss over Time (Case 1)')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.show()

# =============================================================================
# Testing the Agent on Case 1
# =============================================================================

def test(env, num_episodes=40):
    """
    Tests the trained agent on the given environment.
    """
    success_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        elif isinstance(state, np.ndarray):
            state = state.item()
        state_encoded = one_hot_encode(state, state_size)
        done = False
        trajectory = []
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_encoded).to(device)
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            elif isinstance(next_state, np.ndarray):
                next_state = next_state.item()
            row, col = divmod(next_state, env.env.ncol)
            trajectory.append((row, col))
            state_encoded = one_hot_encode(next_state, state_size)
        if reward == 5:  # Adjusted reward for reaching the goal
            success = True
            success_count += 1
        else:
            success = False
        print(f"Episode {episode + 1}: {'Success' if success else 'Failure'}")
        print(f"Trajectory: {trajectory}")
    success_rate = (success_count / num_episodes) * 100
    print(f"Overall Success Rate on Case: {success_rate:.2f}%")

# Test the agent on Case 1
print("\nTesting on Case 1 after adjustments:")
test(env_case1)

# Save the model parameters for Case 1
torch.save(policy_net.state_dict(), 'Konda-model-case1.pt')
print("Model parameters saved to 'Konda-model-case1.pt'")

# =============================================================================
# Training the Agent for Case 2
# =============================================================================

# Adjusted Hyperparameters for Case 2 (if needed)
state_size = env_case2.observation_space.n  # Number of states
action_size = env_case2.action_space.n      # Number of actions
hidden_size = 256                           # Hidden layer size
batch_size = 64                             # Batch size for training
gamma = 0.9                                 # Discount factor
epsilon = 1.0                               # Initial exploration rate
epsilon_min = 0.01                          # Minimum exploration rate
epsilon_decay = 0.995                       # Exploration decay rate
learning_rate = 0.01                        # Learning rate
target_update = 10                          # Frequency to update target network
num_episodes = 3000                         # Number of training episodes

# Re-initialize the policy network
policy_net = DQN(state_size, action_size, hidden_size).to(device)

# Re-initialize the target network
target_net = DQN(state_size, action_size, hidden_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Re-initialize the optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Re-initialize the replay memory
memory = ReplayMemory(50000)

# Define the loss function
criterion = nn.MSELoss()

# Reset epsilon
epsilon = 1.0

# Train the agent on Case 2
print("\nTraining on Case 2...")
rewards_case2, losses_case2 = train(env_case2, num_episodes)

# Plot the total rewards over episodes
plt.figure(figsize=(12, 6))
plt.plot(rewards_case2)
plt.title('Rewards over Episodes (Case 2)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Plot the loss over time
plt.figure(figsize=(12, 6))
plt.plot(losses_case2)
plt.title('Loss over Time (Case 2)')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.show()

# =============================================================================
# Testing the Agent on Case 2
# =============================================================================

# Test the agent on Case 2
print("\nTesting on Case 2:")
test(env_case2)

# Save the model parameters for Case 2
torch.save(policy_net.state_dict(), 'Konda-model-case2.pt')
print("Model parameters saved to 'Konda-model-case2.pt'")

# =============================================================================
# Conclusion
# =============================================================================

print("\nTraining and testing completed for both Case 1 and Case 2.")