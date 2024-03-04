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
    def __init__(self, model, model_baseline, stream_factory, actions, num_samples_per_epoch, num_epochs_per_episode):
        self.model = model
        self.model_baseline = model_baseline

        self.current_episode = 0
        self.stream_factory = stream_factory
        self.stream = stream_factory.create(self.current_episode)

        self.state = None  # Initialize state - 0 indicates no change in accuracy

        self.actions = actions

        self.current_epoch = 0
        self.num_epochs = num_epochs_per_episode
        self.num_samples_per_epoch = num_samples_per_epoch
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []


    def reset(self):
        # Reset the environment
        self.current_episode += 1
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []
        self.current_epoch = 0
        self.state = None # Initialize state - 0 indicates no change in accuracy
        self.stream = self.stream_factory.create(seed=self.current_episode)

        # Return the initial state
        return self.state

    def step(self, action):
        # Update delta_hard based on the chosen action
        self.model.delta = self.update_delta_hard(action)

        # Run one epoch of the experiment
        accuracy, reward = self.run_one_epoch()

        # Increment the epoch counter
        self.current_epoch += 1

        # Compute accuracy change of current epoch from the last epoch
        epoch_accuracy_change = 0
        if self.last_accuracy is not None:
            epoch_accuracy_change = ((accuracy - self.last_accuracy) / self.last_accuracy)

        # Update the last accuracy
        self.last_accuracy = accuracy

        # Update the list of last 5 epoch accuracies
        self.last_5_epoch_accuracies.append(accuracy)
        if len(self.last_5_epoch_accuracies) > 5:
            self.last_5_epoch_accuracies.pop(0)

        # Compute accuracy change of current epoch over the average of last 5 epochs
        epoch_5_accuracy_change = 0
        if len(self.last_5_epoch_accuracies) == 5:
            #epoch_5_accuracy_change = ((accuracy - self.last_5_epoch_accuracies[0]) / self.last_5_epoch_accuracies[0]) * 100
            epoch_5_accuracy_change = ((accuracy - np.mean(self.last_5_epoch_accuracies)) / np.mean(self.last_5_epoch_accuracies))

        # Bin the accuracy changes
        epoch_accuracy_change_bin = self.bin_accuracy_change(epoch_accuracy_change)
        epoch_5_accuracy_change_bin = self.bin_accuracy_change(epoch_5_accuracy_change)

        # Update the state with the accuracy changes
        self.state = self.index_state(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin)

        # Signal if the episode is done
        done = self.current_epoch == self.num_epochs

        return self.state, reward, done

    @staticmethod
    def bin_accuracy_change(accuracy_change):
        if accuracy_change <= (5.0/100.0):
            return 1
        elif accuracy_change <= (10.0/100.0):
            return 2
        elif accuracy_change <= (25.0/100.0):
            return 3
        elif accuracy_change <= (50.0/100.0):
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

        delta = self.model.delta * (action)

        # Ensure delta_hard is within the range [1e-10, 1]
        if delta < 1e-10:
            return 1e-10
        elif delta > 1:
            return 1

        return delta

    
    def run_one_epoch(self):
        # Initialize the total accuracy for this epoch
        total_correctly_classified = 0
        total_baseline_correctly_classified = 0
        total_samples = 0

        # Iterate over the data in the stream
        for x, y in self.stream.take(self.num_samples_per_epoch):

            # Predict the output for the current input
            prediction = self.model.predict_one(x)
            baseline_prediction = self.model_baseline.predict_one(x)

            # Learn from the current input-output pair
            self.model.learn_one(x, y)
            self.model_baseline.learn_one(x, y)

            # Calculate the accuracy of the prediction against the actual output
            is_correctly_classified = self.correctly_classified(prediction, y)
            is_baseline_correctly_classified = self.correctly_classified(baseline_prediction, y)

            # Add the accuracy to the total accuracy
            total_correctly_classified += is_correctly_classified
            total_baseline_correctly_classified += is_baseline_correctly_classified

            # Increment the total number of samples
            total_samples += 1

        # Calculate the prequential accuracy for this epoch
        epoch_prequential_accuracy = self.average_classification_accuracy(total_correctly_classified, total_samples)
        baseline_epoch_prequential_accuracy = self.average_classification_accuracy(total_baseline_correctly_classified, total_samples)

        # Calculate the reward
        reward = epoch_prequential_accuracy - baseline_epoch_prequential_accuracy

        # Return the prequential accuracy and reward
        return epoch_prequential_accuracy, reward

    @staticmethod
    def average_classification_accuracy(correct_predictions, total_predictions):
        return correct_predictions / total_predictions if total_predictions else 0

    @staticmethod
    def correctly_classified(prediction, actual):
        # Check if the prediction is correct
        is_correct = prediction == actual

        # Return 1 if the prediction is correct, 0 otherwise
        return 1 if is_correct else 0


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
        if state is None:  # Skip the iteration if state is None
            return
        best_next_action = np.argmax(self.Q_table[next_state])
        td_target = reward + self.gamma * self.Q_table[next_state][best_next_action]
        td_error = td_target - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error
        print (state, action, reward)


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

    def update_Q_values(self, episode):
        # Update Q-values based on observed episode returns
        returns = 0
        # Iterate over the episode in reverse order to calculate returns
        # Iterate until the first state-action pair is reached where state is None
        # Ensure that the first state-action pair is not included in the iteration

        for i in reversed(range(len(episode))):  
            state, action, reward = episode[i]
            if state is None:  # Skip the iteration if state is None
                continue
            returns = reward + self.gamma * returns  # Calculate discounted return
            print (state, action, reward, returns)
            self.visits[state][action] += 1  # Increment visit count for the state-action pair
            alpha = 1 / self.visits[state][action]  # Step size (adaptive)
            self.Q_table[state][action] += alpha * (returns - self.Q_table[state][action])  # Update Q-value


###################


def train_agent(agent, env, num_episodes):
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # QLearningAgent requires the next state to update Q-values
            if isinstance(agent, QLearningAgent):
                agent.update_Q_values(state, action, reward, next_state)
            # MonteCarloAgent requires the entire episode to update Q-values
            elif isinstance(agent, MonteCarloAgent):
                agent.episode.append((state, action, reward))
            state = next_state
        
        # If the agent is a MonteCarloAgent, update Q-values after the episode is complete
        if isinstance(agent, MonteCarloAgent):
            agent.update_Q_values(agent.episode)
            agent.episode = []  # Clear the episode for the next run


#####################

def setup_environment_and_train(agent_class, agent_name):
    # Since CONFIG and other required variables are not defined in this snippet, 
    # they should be defined elsewhere in the code or passed as arguments to the function.

    # Setup stream factory
    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])

    # Setup model
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])
    model_baseline = ModelClass(delta=CONFIG['delta_hard'])

    # Setup Actions
    actions = CONFIG['actions']['delta_move']

    # Setup Environment
    num_samples_per_epoch = CONFIG['evaluation_interval']
    num_epochs = CONFIG['num_epochs']
    num_states = 25  # Number of possible state combinations

    # Train agent
    env = Environment(model, model_baseline, stream_factory, actions, num_samples_per_epoch, num_epochs)
    agent = agent_class(num_states=num_states, num_actions=len(actions))

    # Add an empty list for episodes if the agent is MonteCarloAgent
    if isinstance(agent, MonteCarloAgent):
        agent.episode = []
    
    train_agent(agent, env, num_episodes=10)

    print(f"Q-table ({agent_name}):")
    # print only 4 significant digits
    np.set_printoptions(precision=4)
    print(agent.Q_table)

def main():
    random.seed(CONFIG['seed0'])
    np.random.seed(CONFIG['seed0'])

    # Train Monte Carlo agent
    setup_environment_and_train(MonteCarloAgent, "Monte Carlo")

    # Train Q-learning agent
    setup_environment_and_train(QLearningAgent, "Q-learning")


if __name__ == "__main__":
    main()
