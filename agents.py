import numpy as np

class BaseAgent:
    def __init__(self, num_states, num_actions, config):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = config['epsilon']  # Epsilon for epsilon-greedy policy
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-values to zeros
        self.rng = np.random.default_rng(0)

    def select_action(self, state, context=None):
        # Select action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
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
    def __init__(self, num_states, num_actions, config):
        super().__init__(num_states, num_actions, config)
        self.alpha = config['alpha']  # Learning rate
        self.gamma = config['gamma']  # Discount factor
        self.alpha_decay = config['alpha_decay']
        # Uses the Q_table initialized in the BaseAgent class

    def select_action(self, state, context=None):
        self.alpha *= self.alpha_decay  # Decay learning rate
        return super().select_action(state, context)

class MonteCarloAgent(BaseAgent):
    def __init__(self, num_states, num_actions, config):
        super().__init__(num_states, num_actions, config)
        self.gamma = config['gamma']  # Discount factor
        self.visits = np.zeros((num_states, num_actions))  # Track number of visits for each state-action pair
        # Uses the Q_table from the BaseAgent class

