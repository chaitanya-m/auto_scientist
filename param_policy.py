import random
import numpy as np
from river import tree
from river.datasets.synth import RandomRBF, RandomTree, ConceptDriftStream
from collections import OrderedDict


# CONSTANTS
CONFIG = {
    'change_point_epoch': 10,
    'evaluation_interval': 1000,
    'num_epochs': 20,
    'delta_easy': 1e-3,
    'delta_hard': 1e-7,
    'seed0': 0,
    'seed1': 100,
    'update_delta_dropped_accuracy': 1.0,
    'num_runs': 2,
    'model': 'UpdatableHoeffdingTreeClassifier',
    'stream_type': 'RandomTree',
    'streams': {
        'RandomTree': {
            'n_classes': 3,
            'n_num_features': 3,
            'n_cat_features': 3,
            'n_categories_per_feature': 3,
            'max_tree_depth': 5,
            'first_leaf_level': 3,
            'fraction_leaves_per_level': 0.15,
        },
    },
    'actions': {
        # Actions to change delta_hard are a multiplier list from 1/100 to 100, with 1 meaning no change
        'delta_move':  [1/100, 1/10, 1, 10, 100],
    }
}

class Environment:
    def __init__(self, model, stream, actions, num_samples_per_epoch, num_epochs_per_episode):
        self.model = model
        self.stream = stream
        self.current_epoch = 0
        self.num_epochs = num_epochs_per_episode
        self.num_samples_per_epoch = num_samples_per_epoch
        self.actions = actions
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []
        self.state = 0  # Initialize state - 0 indicates no change in accuracy

    def reset(self):
        # Reset the environment
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []
        self.current_epoch = 0
        self.state = 0 # Initialize state - 0 indicates no change in accuracy

        # Return the initial state
        return self.state

    def step(self, action_idx):
        # Run one epoch of the experiment
        accuracy = self.run_one_epoch()

        # Increment the epoch counter
        self.current_epoch += 1

        # Compute accuracy change of current epoch from the last epoch
        epoch_accuracy_change = 0
        if self.last_accuracy is not None:
            epoch_accuracy_change = ((accuracy - self.last_accuracy) / self.last_accuracy) * 100

        # Update the last accuracy
        self.last_accuracy = accuracy

        # Update the list of last 5 epoch accuracies
        self.last_5_epoch_accuracies.append(accuracy)
        if len(self.last_5_epoch_accuracies) > 5:
            self.last_5_epoch_accuracies.pop(0)

        # Compute accuracy change of current epoch over the last 5 epochs
        epoch_5_accuracy_change = 0
        if len(self.last_5_epoch_accuracies) == 5:
            epoch_5_accuracy_change = ((accuracy - self.last_5_epoch_accuracies[0]) / self.last_5_epoch_accuracies[0]) * 100

        # Bin the accuracy changes
        epoch_accuracy_change_bin = self.bin_accuracy_change(epoch_accuracy_change)
        epoch_5_accuracy_change_bin = self.bin_accuracy_change(epoch_5_accuracy_change)

        # Update the state with the accuracy changes
        self.state = self.index_state(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin)

        # Compute the reward
        reward = self.compute_reward(accuracy)

        # When the experiment is done, return the Q table
        done = self.current_epoch == self.num_epochs

        return self.state, reward, done

    @staticmethod
    def bin_accuracy_change(accuracy_change):
        if accuracy_change < 5:
            return 1
        elif 5 <= accuracy_change < 10:
            return 2
        elif 10 <= accuracy_change < 25:
            return 3
        elif 25 <= accuracy_change < 50:
            return 4
        else:
            return 5

    @staticmethod
    def index_state(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin):
        # We take epoch_accuracy_change_bin and epoch_5_accuracy_change_bin and return a state index in the range [0, 24]
        return (epoch_accuracy_change_bin - 1) * 5 + (epoch_5_accuracy_change_bin - 1)

    def update_delta_hard(self, action_idx):
        # Adjust delta_hard based on the chosen action index
        action = self.actions[action_idx]
        return max(0, self.model.delta ** (-action))

    
    def run_one_epoch(self):
        # Initialize the total accuracy for this epoch
        total_correctly_classified = 0
        total_samples = 0

        # Iterate over the data in the stream
        for x, y in self.stream.take(self.num_samples_per_epoch):
            # Predict the output for the current input
            prediction = self.model.predict_one(x)

            # Learn from the current input-output pair
            self.model.learn_one(x, y)

            # Calculate the accuracy of the prediction against the actual output
            is_correctly_classified = self.correctly_classified(prediction, y)

            # Add the accuracy to the total accuracy
            total_correctly_classified += is_correctly_classified

            # Increment the total number of samples
            total_samples += 1

        # Calculate the prequential accuracy for this epoch
        epoch_prequential_accuracy = self.average_classification_accuracy(total_correctly_classified, total_samples)

        # Return the prequential accuracy
        return epoch_prequential_accuracy

    @staticmethod
    def average_classification_accuracy(correct_predictions, total_predictions):
        return correct_predictions / total_predictions if total_predictions else 0

    @staticmethod
    def correctly_classified(prediction, actual):
        # Check if the prediction is correct
        is_correct = prediction == actual

        # Return 1 if the prediction is correct, 0 otherwise
        return 1 if is_correct else 0

    def compute_reward(self, accuracy):
        # Compute the change in accuracy from the last epoch
        reward = 0
        if self.last_accuracy is not None:
            reward = accuracy - self.last_accuracy
        self.last_accuracy = accuracy
        return reward


class StreamFactory:
    def __init__(self, stream_type, preinitialized_params):
        self.stream_type = stream_type
        self.preinitialized_params = preinitialized_params

    def create(self, seed):
        if self.stream_type == 'RandomTree':
            return RandomTree(seed_tree=seed, seed_sample=seed, **self.preinitialized_params)
        elif self.stream_type == 'RandomRBF':
            return RandomRBF(seed_model=seed, seed_sample=seed, **self.preinitialized_params)
        else:
            raise ValueError(f"Unknown stream type: {self.stream_type}")


class UpdatableHoeffdingTreeClassifier(tree.HoeffdingTreeClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta


class UpdatableEFDTClassifier(tree.ExtremelyFastDecisionTreeClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta


# Create a dictionary mapping class names to class objects
model_classes = {
    'UpdatableHoeffdingTreeClassifier': UpdatableHoeffdingTreeClassifier,
    'UpdatableEFDTClassifier': UpdatableEFDTClassifier,
}

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

    def update_Q_values(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.Q_table[next_state])
        td_target = reward + self.gamma * self.Q_table[next_state][best_next_action]
        td_error = td_target - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error

    def train(self, env, num_episodes):
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.update_Q_values(state, action, reward, next_state)
                state = next_state


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
        if state is None or np.random.rand() < self.epsilon:
            # If state is None or with probability epsilon, return a random action
            return np.random.randint(self.num_actions)
        else:
            # Select action greedily based on current Q-values for the given state
            return np.argmax(self.Q_table[state])

    def update_Q_values(self, episode):
        # Update Q-values based on observed episode returns
        returns = 0
        for i in reversed(range(len(episode))):  # Iterate over the episode in reverse order to calculate returns
            state, action, reward = episode[i]
            returns = reward + self.gamma * returns  # Calculate discounted return
            print (state, action, reward, returns)
            self.visits[state][action] += 1  # Increment visit count for the state-action pair
            alpha = 1 / self.visits[state][action]  # Step size (adaptive)
            self.Q_table[state][action] += alpha * (returns - self.Q_table[state][action])  # Update Q-value

    def train(self, env, num_episodes):
        # Train the agent by interacting with the environment for a specified number of episodes
        for _ in range(num_episodes):
            episode = []  # Initialize an empty list to store episode transitions
            state = env.reset()  # Reset the environment and get initial state
            done = False  # Flag to indicate whether the episode has terminated
            while not done:
                action = self.select_action(state)  # Select action using the current policy
                next_state, reward, done = env.step(action)  # Take action and observe next state and reward
                episode.append((state, action, reward))  # Store state-action-reward tuple
                state = next_state  # Update current state
            self.update_Q_values(episode)  # Update Q-values based on the observed episode



def main():
    random.seed(CONFIG['seed0'])
    np.random.seed(CONFIG['seed0'])

    # Setup stream factory
    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])
    stream = stream_factory.create(seed=CONFIG['seed0'])

    # Setup model
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])

    # Setup Actions
    actions = CONFIG['actions']['delta_move']

    # Setup Environment
    num_samples_per_epoch = CONFIG['evaluation_interval']
    num_epochs = CONFIG['num_epochs']
    env = Environment(model, stream, actions, num_samples_per_epoch, num_epochs)

    num_states = 25  # Number of possible state combinations

    # Train Monte Carlo agent
    agent_mc = MonteCarloAgent(num_states=num_states, num_actions=len(actions))
    agent_mc.train(env, num_episodes=10)

    print("Q-table (Monte Carlo):")
    print(agent_mc.Q_table)

    # Train Q-learning agent
    env = Environment(model, stream, actions, num_samples_per_epoch, num_epochs)
    agent_q_learning = QLearningAgent(num_states=num_states, num_actions=len(actions))
    agent_q_learning.train(env, num_episodes=10)

    print("Q-table (Q-learning):")
    print(agent_q_learning.Q_table)

    # Compare Q tables
    print("\nQ-table (Q-learning):")
    print(agent_q_learning.Q_table)
    print("\nQ-table (Monte Carlo):")
    print(agent_mc.Q_table)


if __name__ == "__main__":
    main()
