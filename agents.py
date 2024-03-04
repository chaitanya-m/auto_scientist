import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-values to zeros

    def select_action(self, state):
        if state is None or np.random.rand() < self.epsilon:
            # If state is None or with probability epsilon, return a random action
            return np.random.randint(self.num_actions)
        else:
            # Select action greedily based on current Q-values for the given state
            return np.argmax(self.Q_table[state])


class MonteCarloAgent:
    def __init__(self, num_states, num_actions, gamma=0.9, epsilon=0.1):
        # Initialize Monte Carlo agent with the number of states, number of actions, and discount factor gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table and visit counts for each state-action pair
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-values to zero
        self.visits = np.zeros((num_states, num_actions))   # Track number of visits for each state-action pair

    def select_action(self, state):
        if state == None or np.random.rand() < self.epsilon:
            # If state is None or with probability epsilon, return a random action
            return np.random.randint(self.num_actions)
        else:
            # Select action greedily based on current Q-values for the given state
            return np.argmax(self.Q_table[state])
