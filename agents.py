import numpy as np

class BaseAgent:
    def __init__(self, num_states, num_actions, test_phase_length, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-values to zeros
        self.rng = np.random.default_rng(0)
        self.is_test_phase = False
        self.test_phase_counter = 0
        self.test_phase_length = test_phase_length

    def select_action(self, state):
        # Select action using epsilon-greedy policy

        if self.is_test_phase:
            if self.test_phase_counter < self.test_phase_length - 1:
                # If within test phase
                action_no = self.num_actions - 1 # Select the null action
                self.test_phase_counter += 1 # Increment the test phase counter

            elif self.test_phase_counter == self.test_phase_length - 1:
                # If at the end of the test phase
                action_no = self.num_actions - 1 # Select the null action
                self.is_test_phase = False # End the test phase
                self.test_phase_counter = 0 # Reset the test phase counter

        else: # a new test phase will begin if a non null action is selected
            if np.random.rand() < self.epsilon:
                # With probability epsilon, return a random action
                action_no = self.rng.integers(self.num_actions)
                # print("Initial or exploratory action " + str(action_no))

            elif np.amax(self.Q_table[state]) == np.amin(self.Q_table[state]): # If all Q-values are equal, pick a random action
                action_no = self.rng.integers(self.num_actions)
            # print("Actions all still equal " + str(action_no))

            else:
                # Otherwise, select action greedily based on current Q-values for the given state
                action_no = np.argmax(self.Q_table[state])
                # print("Best Q-table action " + str(action_no))

            if action_no < self.num_actions - 1: # If action picked not the last action, ie not the null action... a state change occurs and a new test phase begins
                self.test_phase_counter = 0
                self.is_test_phase = True

        return action_no, self.is_test_phase, self.test_phase_counter

class QLearningAgent(BaseAgent):
    def __init__(self, num_states, num_actions, test_phase_length, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_states, num_actions, epsilon)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.test_phase_length = test_phase_length
        # Uses the Q_table initialized in the BaseAgent class

class MonteCarloAgent(BaseAgent):
    def __init__(self, num_states, num_actions, test_phase_length, alpha = None, gamma=0.9, epsilon=0.1): # Never use alpha for Monte Carlo... this is just to keep the same function signature
        super().__init__(num_states, num_actions, epsilon)
        self.gamma = gamma  # Discount factor
        self.visits = np.zeros((num_states, num_actions))  # Track number of visits for each state-action pair
        self.test_phase_length = test_phase_length
        # Uses the Q_table from the BaseAgent class

