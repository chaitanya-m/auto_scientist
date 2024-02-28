# Import necessary libraries
import concurrent.futures
import queue
import threading
from collections import OrderedDict

from river.datasets.synth import RandomRBF, RandomTree, ConceptDriftStream
from river import tree

import pandas as pd
import random
import numpy as np

from multiprocessing import Pool


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
        # 'RandomRBF': {
        #     'n_classes': 3,
        #     'n_features': 2,
        #     'n_centroids': 3,
        # },
        'RandomTree': {
            'n_classes': 3,
            'n_num_features': 3,
            'n_cat_features': 3,
            'n_categories_per_feature': 3,
            'max_tree_depth': 5,
            'first_leaf_level': 3,
            'fraction_leaves_per_level': 0.15,
        },
        # Add more streams here
    }
}


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

class Experiment:
    def __init__(self, config, model, stream, num_seeds):
        self.config = config
        self.model = model
        self.stream = stream
        self.num_seeds = num_seeds
        self.preq_accuracy = []


    def average_classification_accuracy(correct_predictions, total_predictions):
        return correct_predictions / total_predictions if total_predictions else 0

    def classification_accuracy(self, prediction, actual):
        # Check if the prediction is correct
        is_correct = prediction == actual

        # Return the accuracy (1 if the prediction is correct, 0 otherwise)
        return 1 if is_correct else 0

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
            is_correctly_classified = self.classification_accuracy(prediction, y)

            # Add the accuracy to the total accuracy
            total_correctly_classified += is_correctly_classified

            # Increment the total number of samples
            total_samples += 1

        # Calculate the prequential accuracy for this epoch
        epoch_prequential_accuracy = total_correctly_classified / total_samples

        # Update the prequential accuracy list
        self.preq_accuracy.append(epoch_prequential_accuracy)

        # Return the prequential accuracy
        return epoch_prequential_accuracy

    def ensure_bounds(self, config):
        # Check if the configuration values are within valid bounds
        # Define the valid bounds for each parameter
        is_within_bounds = 0 <= config['delta_easy'] <= 1 and 0 <= config['update_delta_dropped_accuracy'] <= 1
        # Return whether both parameters are within their valid bounds
        return is_within_bounds

class Agent:
    def __init__(self, config, model, stream_factory, exploration, num_episodes, num_seeds):
        self.config = CONFIG
        self.exploration = exploration
        self.num_episodes = num_episodes
        self.num_seeds = num_seeds
        self.model = model
        self.stream_factory = stream_factory
        self.Q_table = {}
        self.Q_table_visits = {}


    def find_best_strategy(self):
        """
        Finds the best strategy for the agent by running experiments on a set of concept drift streams.

        The agent first creates a Q-table where each state is the relative change in average accuracy over the last 10 epochs,
        and each action is an adjustment for the parameters.

        The agent then runs experiments on a set of concept drift streams. For each stream, the agent runs two experiments:
        one without any intervention, and one with intervention where the agent is allowed to adjust the parameters.

        After running the experiments, the agent updates the Q-table based on the difference in accuracy between the two strategies.

        If the strategy with agent intervention has a higher accuracy than the strategy without intervention, the agent updates
        the Q-table with the new reward, which is the difference between the two stream prediction accuracies.

        Returns:
            dict: The updated Q-table.
        """

        # Iterate over the number of episodes
        for episode in range(self.num_episodes):

            # Initialize lists to store epoch-wise accuracies
            epochwise_accuracy_without_intervention = []
            epochwise_accuracy_with_intervention = []

            # Create a set of seeded concept drift streams for each episode
            concept_drift_streams = [ConceptDriftStream(
                stream=self.stream_factory.create(seed=self.config['seed0']+i), 
                drift_stream=self.stream_factory.create(seed=self.config['seed1']+i), 
                position=self.config['change_point_epoch']*self.config['evaluation_interval'], 
                seed=self.config['seed0']+1
            ) for i in range(self.num_seeds)]

            # Prepare the arguments for run_experiment without agent intervention
            no_intervention_args = [(stream, False) for stream in concept_drift_streams]

            # Prepare the arguments for run_experiment with agent intervention
            with_intervention_args = [(stream, True) for stream in concept_drift_streams]

            # Create a multiprocessing pool for the experiments without intervention
            with Pool() as pool:
                # Run the experiments without any agent intervention
                epochwise_accuracy_without_intervention = pool.map(self.run_experiment, no_intervention_args)
                # Close the pool and wait for all tasks to finish
                pool.close()
                pool.join()

            # Create a separate multiprocessing pool for the experiments with intervention
            with Pool() as pool:
                # Run the experiments with agent intervention
                epochwise_accuracy_with_intervention = pool.map(self.run_experiment, with_intervention_args)
                # Close the pool and wait for all tasks to finish
                pool.close()
                pool.join()

            # Compute rewards for each seed for this episode
            for i in range(self.num_seeds):
                for epoch in range(self.config['num_epochs']):
                    accuracy = epochwise_accuracy_with_intervention[i][epoch]
                    if accuracy > epochwise_accuracy_without_intervention[i][epoch]:
                        reward = accuracy - epochwise_accuracy_without_intervention[i][epoch]

                        # Compute the state
                        state = (accuracy - np.mean(epochwise_accuracy_with_intervention[i][-10:])) / np.mean(epochwise_accuracy_with_intervention[i][-10:])
                        if state < -0.1:
                            state = 'Drop > 0.1'
                        elif state < -0.01:
                            state = 'Drop 0.01 - 0.1'
                        else:
                            state = 'Drop 0.0 - 0.01'

                        # Update the Q-table based on the state
                        self.Q_table[state] = reward

        return self.Q_table

    def run_experiment(self, args):
        """
        Runs an experiment on a given stream with or without agent intervention.

        If intervention is allowed, the agent can adjust the parameters in real time.

        The agent will adjust the parameters update_delta_dropped_accuracy and delta_easy if the prequential accuracy 
        for the last epoch is lower than the prequential accuracy * update_delta_dropped_accuracy for the last 10 epochs.

        The agent can consult the Q-table for the best action to take, it can also consult the policy.
        The agent can also choose to explore by taking a random action.

        If intervention is not allowed, the agent cannot adjust the parameters in real time.

        Args:
            args (tuple): A tuple containing the stream and a boolean indicating whether or not the agent should intervene.

        Returns:
            float: The accuracy of the model on the stream.
        """
        stream, intervention = args
        experiment = Experiment(self.config, self.model, stream, self.num_seeds)
        accuracy = 0
        accuracies = []

        if intervention:
            # The agent is allowed to adjust the parameters while running the experiment
            for epoch in range(self.config['num_epochs']):
                # Run the model on the stream for one epoch
                accuracy = experiment.run_one_epoch()
                accuracies.append(accuracy)

                # last_10_epoch_accuracy = experiment.preq_accuracy[-10:] if it exists
                # if it hasn't been 10 epochs yet, use as many epochs as are available
                # The code is:

                mean_historical_accuracy = np.mean(experiment.preq_accuracy[-10:] if experiment.preq_accuracy else [0])

                # If the prequential accuracy for the last epoch is lower than the prequential accuracy * update_delta_dropped_accuracy for the last 10 epochs
                if accuracy < mean_historical_accuracy * self.config['update_delta_dropped_accuracy']:
                    # Then the agent will adjust the parameters update_delta_dropped_accuracy and delta_easy
                    action = self.choose_action()
                    self.config = self.apply_action(action, experiment)

        else:
            # The agent is not allowed to adjust the parameters while running the experiment
            for epoch in range(self.config['num_epochs']):
                # Run the model on the stream for one epoch
                accuracy = experiment.run_one_epoch()
                accuracies.append(accuracy)

        # Return the accuracy of the model on the stream
        return accuracies


    def choose_action(self):
        action = {param: np.random.choice([-step['coarse'], -step['fine'], step['fine'], step['coarse']])
                  for param, step in self.exploration.items()}
        return action

    def apply_action(self, action, experiment):
        # Iterate over each parameter and step size in the action
        for param, step in action.items():
            # Get the adjustment type for the parameter
            adjustment_type = self.exploration[param]['adjustment_type']

            # Define the actions
            actions = {
                'multiplicative': {
                    'positive': lambda x, y: x * y,
                    'negative': lambda x, y: x / y
                },
                'additive': {
                    'positive': lambda x, y: x + y,
                    'negative': lambda x, y: x - y
                }
            }

            # Determine the direction of the action (positive or negative)
            direction = 'positive' if step > 0 else 'negative'

            # Calculate the new parameter value by applying the appropriate lambda function in the actions dictionary
            new_param_value = actions[adjustment_type][direction](self.config[param], abs(step))

            # Ensure the new parameter value is within its valid range
            if experiment.ensure_bounds(self.config): # this is terrible interface design, change this
                # If it is, apply the action
                self.config[param] = new_param_value

        # Return the updated configuration
        return self.config

    def update_policy(self):
        # Initialize the policy
        policy = {}

        # Iterate over the states in the Q-table
        for state, action in self.Q_table.keys():
            # If the state is not in the policy or the reward for the current action is greater than the reward for the current best action
            if state not in policy or self.Q_table[(state, action)] > self.Q_table[(state, policy[state])]:
                # Update the policy with the current action
                policy[state] = action

        return policy

def main():
    
    random.seed(CONFIG['seed0'])
    np.random.seed(CONFIG['seed0'])

    # Setup stream factory
    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])

    # Setup model
    # Use the dictionary to get the class object
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])

    # Setup and run Agent
    exploration = {
        'delta_easy': {'coarse': 100, 'fine': 10,  'adjustment_type': 'multiplicative'},
        'update_delta_dropped_accuracy': {'coarse': 0.1, 'fine': 0.01, 'adjustment_type': 'additive'},
    }
    agent = Agent(CONFIG, model, stream_factory, exploration, num_episodes=10, num_seeds=5)
    policy = agent.find_best_strategy()
    print(policy)

    # Now the policy created by the agent can be tested on a set of new streams using similarity measures for the states
    # The policy can be used to adjust the parameters of the model in real time
    # A comparison can be made between the performance of the model with and without the agent intervention

if __name__ == "__main__":
    main()

