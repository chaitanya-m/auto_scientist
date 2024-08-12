import numpy as np

class BaseAgent:
    def __init__(self, num_states, num_actions, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-values to zeros
        self.rng = np.random.default_rng(0)

    def select_action(self, state, action_delay, action_delay_counter):
        # Select action using epsilon-greedy policy

        if action_delay_counter < action_delay:
            # If action delay is not yet reached, return the last action, as it is always the null action
            action_no = self.num_actions - 1

        elif np.random.rand() < self.epsilon:
            # With probability epsilon, return a random action
            action_no = self.rng.integers(self.num_actions)
            # print("Initial or exploratory action " + str(action_no))

        elif np.amax(self.Q_table[state]) == np.amin(self.Q_table[state]): # If all Q-values are equal
            action_no = self.rng.integers(self.num_actions)
            # print("Actions all still equal " + str(action_no))

        else:
            # Otherwise, select action greedily based on current Q-values for the given state
            action_no = np.argmax(self.Q_table[state])
            # print("Best Q-table action " + str(action_no))
        return action_no

class QLearningAgent(BaseAgent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_states, num_actions, epsilon)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        # Uses the Q_table initialized in the BaseAgent class

class MonteCarloAgent(BaseAgent):
    def __init__(self, num_states, num_actions, alpha = None, gamma=0.9, epsilon=0.1): # Never use alpha for Monte Carlo... this is just to keep the same function signature
        super().__init__(num_states, num_actions, epsilon)
        self.gamma = gamma  # Discount factor
        self.visits = np.zeros((num_states, num_actions))  # Track number of visits for each state-action pair
        # Uses the Q_table from the BaseAgent class

