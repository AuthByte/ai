import gym
import numpy as np

# Create the Lunar Lander environment
env = gym.make('LunarLander-v2')

# Hyperparameters
num_episodes = 1000
max_steps = 200
learning_rate = 0.01
discount_factor = 0.99

# Initialize Q-table
q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))

# Function to choose action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = max(0.1, 1.0 - episode / (num_episodes / 2))  # Decaying epsilon

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        # Render the environment
        env.render()
        
        # Update Q-table using the Q-learning formula
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
        total_reward += reward
        
        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
