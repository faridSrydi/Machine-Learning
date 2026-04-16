import numpy as np

# Definisi environment (maze 3x3)
# 0 = jalan, 1 = tembok, 2 = goal
maze = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 2]
])

# Inisialisasi Q-table (states x actions)
q_table = np.zeros((9, 4))  # 9 states (3x3), 4 actions (up, down, left, right)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Helper functions
def state_to_pos(state):
    return divmod(state, maze.shape[1])


def pos_to_state(pos):
    return pos[0] * maze.shape[1] + pos[1]


def is_valid(pos):
    if pos[0] < 0 or pos[0] >= maze.shape[0] or pos[1] < 0 or pos[1] >= maze.shape[1]:
        return False
    if maze[pos] == 1:
        return False
    return True


def take_action(state, action):
    r, c = state_to_pos(state)
    if action == 0:  # up
        next_pos = (r - 1, c)
    elif action == 1:  # down
        next_pos = (r + 1, c)
    elif action == 2:  # left
        next_pos = (r, c - 1)
    else:  # right
        next_pos = (r, c + 1)

    if not is_valid(next_pos):
        return state, -1, False

    next_state = pos_to_state(next_pos)
    if maze[next_pos] == 2:  # goal
        return next_state, 10, True

    return next_state, -0.1, False


# Training loop
for episode in range(1000):
    state = 0  # Start at (0,0)
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(q_table[state])

        # Execute action
        next_state, reward, done = take_action(state, action)

        # Update Q-table
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

# Print learned Q-values
print("Learned Q-table:")
print(q_table)
