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
    }
}


class Agent:
    def __init__(self, exploration):
        self.exploration = exploration
        self.Q_table = {}

    def choose_action(self, state):
        # Here, you would typically use the state to decide on the action
        # For now, let's choose a random action
        action = {param: np.random.choice([-step['coarse'], -step['fine'], step['fine'], step['coarse']])
                  for param, step in self.exploration.items()}
        return action

    def update_q_table(self, state, action, accuracy):
        # Update the Q-table based on the state, action, and accuracy
        # For now, let's update the Q-table with a random value
        self.Q_table[(state, tuple(action))] = np.random.random()


class Environment:
    def __init__(self, config, model, stream, agent):
        self.config = config
        self.model = model
        self.stream = stream
        self.agent = agent
        self.stream = None
        self.current_epoch = 0

    def reset(self):
        # Reset the environment
        pass

    def step(self):
        # Run one epoch of the experiment
        accuracy = self.run_one_epoch()

        # Increment the epoch counter
        self.current_epoch += 1

        # Update the state with the new accuracy
        state = {'accuracy': accuracy}

        # Let the agent choose an action based on the state
        action = self.agent.choose_action(state)

        # Update the Q table
        self.agent.update_q_table(state, action, accuracy)

        # Compute the reward
        reward = self.compute_reward(accuracy)

        # When the experiment is done, return the Q table
        done = self.current_epoch == self.config['num_epochs']
        return state, reward, done

    def run_one_epoch(self):
        # Initialize the total accuracy for this epoch
        total_correctly_classified = 0
        total_samples = 0

        # Iterate over the data in the stream
        for x, y in self.stream.take(self.config['evaluation_interval']):
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
        # Example reward function: higher accuracy leads to higher reward
        return accuracy


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
    # Add more classes as needed
}


def main():
    random.seed(CONFIG['seed0'])
    np.random.seed(CONFIG['seed0'])

    # Setup stream factory
    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])
    stream = stream_factory.create(CONFIG['seed0'])

    # Setup model
    # Use the dictionary to get the class object
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])

    # Setup Agent
    exploration = {
        'delta_easy': {'coarse': 100, 'fine': 10, 'adjustment_type': 'multiplicative'},
        'update_delta_dropped_accuracy': {'coarse': 0.1, 'fine': 0.01, 'adjustment_type': 'additive'},
    }
    agent = Agent(exploration)

    # Setup Environment
    env = Environment(CONFIG, model, stream, agent, num_seeds=5)

    # Reset the environment to start a new experiment
    env.reset()

    # Main loop
    done = False
    while not done:
        state, reward, done = env.step()
        print(f"Epoch: {env.current_epoch}, Accuracy: {state['accuracy']}, Reward: {reward}")


if __name__ == "__main__":
    main()



# all the agent is returning is an action
    
# step function: all the env is returning is the next state and reward