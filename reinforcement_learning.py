import random
import numpy as np
import gym
from matplotlib import pyplot as plt
import math
import torch

random.seed(10)
class MyEnvironment:

    def __init__(self):
        self.table = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, -1, 1, -1, 1, 3, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
        self.terminal_states = [(2, 1), (2, 3), (2, 5)]
        self.num_actions = 4  # Discrete actions: 0 (up), 1 (down), 2 (left), 3 (right)
        non_ones = 0
        for row in self.table:
            for element in row:
                if element != 1:
                    non_ones += 1

        self.num_observations = non_ones
        self.initial_state = (1, 1)            # Initial state of the agent
        self.current_state = self.initial_state

    def reset(self):
        # Reset the environment to its initial state
        self.current_state = self.initial_state
        return self.current_state

    def take_action(self, action):
        prob_rm = 0.6  # Probability of a correct move
        prob_wm = 0.2  # Probability of a wrong move

        if action == 0:
            new_state_r, new_state_c = self.move("up", prob_rm, prob_wm)
        elif action == 1:
            new_state_r, new_state_c = self.move("down", prob_rm, prob_wm)
        elif action == 2:
            new_state_r, new_state_c = self.move("left", prob_rm, prob_wm)
        elif action == 3:
            new_state_r, new_state_c = self.move("right", prob_rm, prob_wm)
        else:
            raise ValueError("Invalid action")

        reward = self.calculate_reward(new_state_r, new_state_c)
        episode_ended = (new_state_r, new_state_c) in self.terminal_states
        self.current_state = (new_state_r, new_state_c)  # Update the current state
        return (new_state_r, new_state_c), reward, episode_ended

    def move(self, direction, prob_rm, prob_wm):
        if random.uniform(0, 1) < prob_rm:
            if not self.hit_wall(direction):
                return getattr(self, f"move_{direction}")()
            else:
                return self.current_state  # Stay in the same state if hitting a wall
        else:
            return self.glide(direction)

    def move_up(self):
        return self.current_state[0] - 1, self.current_state[1]

    def move_down(self):
        return self.current_state[0] + 1, self.current_state[1]

    def move_right(self):
        return self.current_state[0], self.current_state[1] + 1

    def move_left(self):
        return self.current_state[0], self.current_state[1] - 1

    def hit_wall(self, direction):
        if direction == "up" and (self.table[self.current_state[0] - 1][self.current_state[1]] == 1):
            return 1
        elif direction == "down" and (self.table[self.current_state[0] + 1][self.current_state[1]] == 1):
            return 1
        elif direction == "right" and (self.table[self.current_state[0]][self.current_state[1] + 1] == 1):
            return 1
        elif direction == "left" and (self.table[self.current_state[0]][self.current_state[1] - 1] == 1):
            return 1
        else:
            return 0

    def glide(self, direction):
        if direction == "up":
            if random.uniform(0, 1) < 0.5:
                direction = "right"
            else:
                direction = "left"
            if not self.hit_wall(direction):
                return getattr(self, f"move_{direction}")()
            else:
                return self.current_state
        if direction == "right":
            if random.uniform(0, 1) < 0.5:
                direction = "up"
            else:
                direction = "down"
            if not self.hit_wall(direction):
                return getattr(self, f"move_{direction}")()
            else:
                return self.current_state
        if direction == "down":
            if random.uniform(0, 1) < 0.5:
                direction = "right"
            else:
                direction = "left"
            if not self.hit_wall(direction):
                return getattr(self, f"move_{direction}")()
            else:
                return self.current_state
        if direction == "left":
            if random.uniform(0, 1) < 0.5:
                direction = "up"
            else:
                direction = "down"
            if not self.hit_wall(direction):
                return getattr(self, f"move_{direction}")()
            else:
                return self.current_state

    def calculate_reward(self, state_r, state_c):
        num_rows = len(self.table)
        num_cols = len(self.table[0])


        if 0 <= state_r < num_rows and 0 <= state_c < num_cols:
            return self.table[state_r][state_c]
        else:

            return 0


    def conversion(self, current_state):
        if current_state == (1, 1):
            return 0
        if current_state == (1, 2):
            return 1
        if current_state == (1, 3):
            return 2
        if current_state == (1, 4):
            return 3
        if current_state == (1, 5):
            return 4
        if current_state == (2, 1):
            return 5
        if current_state == (2, 3):
            return 6
        if current_state == (2, 5):
            return 7




env = MyEnvironment()
# Q-table initialization

num_actions = env.num_actions
num_observations = env.num_observations
Q_table = np.zeros((num_observations, num_actions))
print(Q_table)

lr = math.log(2)/2 # Learning rate
gamma = 0.9# Discount factor
exploration_proba = 0.1 # Initial exploration probability

n_episodes = 1000
max_iter_episode = 100
rewards_per_episode = []
V_value_0 = []
V_value_1 = []
V_value_2 = []
V_value_3 = []
V_value_4 = []
V_value_5 = []
V_value_6 = []
V_value_7 = []
mean_V_values_per_episode = []
# Q-learning algorithm
for e in range(n_episodes):
    current_state = env.reset()
    done = False
    total_episode_reward = 0
    for i in range(max_iter_episode):

        # Explore or exploit based on exploration probability
        if np.random.uniform(0, 1) < exploration_proba:
            action = np.random.choice(range(num_actions))
        else:
            action = np.argmax(Q_table[env.conversion(current_state), :])

        next_state, reward, done = env.take_action(action)

        Q_table[env.conversion(current_state), action] = (1 - lr) * Q_table[env.conversion(current_state), action] + lr * (
                    reward + gamma * np.max(Q_table[env.conversion(next_state), :]))

        V_value_0.append(np.max(Q_table[0, :]))
        V_value_1.append(np.max(Q_table[1, :]))
        V_value_2.append(np.max(Q_table[2, :]))
        V_value_3.append(np.max(Q_table[3, :]))
        V_value_4.append(np.max(Q_table[4, :]))
        V_value_5.append(np.max(Q_table[5, :]))
        V_value_6.append(np.max(Q_table[6, :]))
        V_value_7.append(np.max(Q_table[7, :]))
        mean_V_values = [np.mean(Q_table[state, :]) for state in range(num_observations)]
        mean_V_values_per_episode.append(mean_V_values)
        total_episode_reward += (gamma**i)*reward
        if done:
            break

        current_state = next_state

    lr = math.log(e+1)/(e+1)
    rewards_per_episode.append(total_episode_reward)
print(rewards_per_episode)
print(Q_table)

num_test_episodes = 10
average_test_reward = 0
gamma_discount =0.999
for _ in range(num_test_episodes):
    current_state = env.reset()
    done = False
    total_episode_reward = 0

    for _ in range(max_iter_episode):

        action = np.argmax(Q_table[env.conversion(current_state), :])

        next_state, reward, done = env.take_action(action)
        total_episode_reward += gamma_discount * reward
        gamma_discount *= gamma

        if done:
            break

        current_state = next_state

    average_test_reward += total_episode_reward

average_test_reward /= num_test_episodes

print(f'Average total reward over {num_test_episodes} test episodes: {average_test_reward}')



plt.figure(1)

plt.plot(V_value_0)
plt.title("State A1")
plt.ylabel("V_values")
plt.figure(2)
plt.plot(V_value_1)
plt.title("State A2")
plt.ylabel("V_values")
plt.figure(3)
plt.plot(V_value_2)
plt.title("State A3")
plt.ylabel("V_values")
plt.figure(4)
plt.title("State A4")
plt.plot(V_value_3)
plt.ylabel("V_values")
plt.figure(5)
plt.plot(V_value_4)
plt.title("State A5")
plt.ylabel("V_values")
plt.figure(6)
plt.show()

#exploration_probability experiment
def run_experiment(exploration_probabilities, n_episodes, max_iter_episode):
    results = []

    for exploration_proba in exploration_probabilities:
        Q_table = np.zeros((num_observations, num_actions))
        V_values_per_episode = np.zeros((n_episodes, num_observations))
        rewards_per_episode = []

        for e in range(n_episodes):
            current_state = env.reset()
            done = False
            total_episode_reward = 0

            for i in range(max_iter_episode):
                if np.random.uniform(0, 1) < exploration_proba:
                    action = np.random.choice(range(num_actions))
                else:
                    action = np.argmax(Q_table[env.conversion(current_state), :])

                next_state, reward, done = env.take_action(action)

                Q_table[env.conversion(current_state), action] = (1 - lr) * Q_table[env.conversion(current_state), action] + lr * (
                            reward + gamma * np.max(Q_table[env.conversion(next_state), :]))

                total_episode_reward += (gamma**i)*reward
                if done:
                    break
                current_state = next_state

            rewards_per_episode.append(total_episode_reward)
            V_values_per_episode[e, :] = [np.max(Q_table[state, :]) for state in range(num_observations)]

        results.append({
            'exploration_probability': exploration_proba,
            'rewards_per_episode': rewards_per_episode,
            'V_values_per_episode': V_values_per_episode,
            'Q_table': Q_table
        })

    return results
lr = math.log(2)/2  # Learning rate
gamma = 0.9
exploration_probabilities = [0.1, 0.2, 0.5, 0.8]
env = MyEnvironment()
num_actions = env.num_actions
num_observations = env.num_observations
n_episodes = 1000
max_iter_episode = 100
experiment_results = run_experiment(exploration_probabilities, n_episodes, max_iter_episode)

plt.figure(figsize=(10, 6))
for result in experiment_results:
    plt.plot(result['rewards_per_episode'], label=f'Exploration Proba: {result["exploration_probability"]}')

plt.title('Learning Curves for Different Exploration Probabilities')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for result in experiment_results:
    mean_V_values = np.mean(result['V_values_per_episode'], axis=1)
    plt.plot(mean_V_values, label=f'Exploration Probability: {result["exploration_probability"]}')

plt.title('Convergence Speed of V-values for Different Exploration Probabilities')
plt.xlabel('Episode')
plt.ylabel('Mean V-value')
plt.legend()
plt.show()

#experiment for learning rate
def run_experiment(learning_rates, n_episodes, max_iter_episode):
    results = []

    for lr in learning_rates:
        Q_table = np.zeros((num_observations, num_actions))
        V_values_per_episode = np.zeros((n_episodes, num_observations))
        rewards_per_episode = []

        for e in range(n_episodes):
            current_state = env.reset()
            done = False
            total_episode_reward = 0

            for i in range(max_iter_episode):
                if np.random.uniform(0, 1) < exploration_proba:
                    action = np.random.choice(range(num_actions))
                else:
                    action = np.argmax(Q_table[env.conversion(current_state), :])

                next_state, reward, done = env.take_action(action)

                Q_table[env.conversion(current_state), action] = (1 - lr) * Q_table[env.conversion(current_state), action] + lr * (
                            reward + gamma * np.max(Q_table[env.conversion(next_state), :]))

                total_episode_reward += (gamma**i)*reward
                if done:
                    break
                current_state = next_state

            rewards_per_episode.append(total_episode_reward)
            V_values_per_episode[e, :] = [np.max(Q_table[state, :]) for state in range(num_observations)]

        results.append({
            'learning_rate': lr,
            'rewards_per_episode': rewards_per_episode,
            'V_values_per_episode': V_values_per_episode,
            'Q_table': Q_table
        })

    return results

gamma = 0.9 # Discount factor
exploration_proba = 0.1  # Exploration probability
learning_rates = [0.01, 0.1, 0.2, 0.5, 0.9]
env = MyEnvironment()
num_actions = env.num_actions
num_observations = env.num_observations

n_episodes = 1000
max_iter_episode = 100
experiment_results = run_experiment(learning_rates, n_episodes, max_iter_episode)
plt.figure(figsize=(10, 6))
for result in experiment_results:
    mean_V_values = np.mean(result['V_values_per_episode'], axis=1)
    plt.plot(mean_V_values, label=f'Learning Rate: {result["learning_rate"]}')

plt.title('Convergence Speed of V-values for Different Learning Rates')
plt.xlabel('Episode')
plt.ylabel('Mean V-value')
plt.legend()
plt.show()

def run_experiment(dynamic_lr, n_episodes, max_iter_episode):
    Q_table = np.zeros((num_observations, num_actions))
    V_values_per_episode = np.zeros((n_episodes, num_observations))
    rewards_per_episode = []

    for e in range(n_episodes):
        current_state = env.reset()
        done = False
        total_episode_reward = 0

        # Calculate dynamic learning rate
        lr = math.log(e+1)/(e+1) if dynamic_lr else 0.2

        for i in range(max_iter_episode):
            if np.random.uniform(0, 1) < exploration_proba:
                action = np.random.choice(range(num_actions))
            else:
                action = np.argmax(Q_table[env.conversion(current_state), :])

            next_state, reward, done = env.take_action(action)

            Q_table[env.conversion(current_state), action] = (1 - lr) * Q_table[env.conversion(current_state), action] + lr * (
                        reward + gamma * np.max(Q_table[env.conversion(next_state), :]))

            total_episode_reward += (gamma**i)*reward
            if done:
                break
            current_state = next_state

        rewards_per_episode.append(total_episode_reward)
        V_values_per_episode[e, :] = [np.max(Q_table[state, :]) for state in range(num_observations)]

    return rewards_per_episode, V_values_per_episode


gamma = 0.9  # Discount factor
exploration_proba = 0.1  # Exploration probability
num_actions = env.num_actions
num_observations = env.num_observations
n_episodes = 1000
max_iter_episode = 100
rewards_const_lr, v_values_const_lr = run_experiment(dynamic_lr=False, n_episodes=n_episodes, max_iter_episode=max_iter_episode)
rewards_dynamic_lr, v_values_dynamic_lr = run_experiment(dynamic_lr=True, n_episodes=n_episodes, max_iter_episode=max_iter_episode)
plt.figure(figsize=(10, 6))
plt.plot(np.mean(v_values_const_lr, axis=1), label='Constant Learning Rate (0.2)')
plt.plot(np.mean(v_values_dynamic_lr, axis=1), label='Dynamic Learning Rate (lr = math.log(e+1)/(e+1))')

plt.title('Convergence Speed of V-values with Different Learning Rates')
plt.xlabel('Episode')
plt.ylabel('Mean V-value')
plt.legend()
plt.show()
