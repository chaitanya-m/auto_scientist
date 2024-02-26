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

# CONSTANTS
CONFIG = {
    'random_seed': 0,
    'change_point': 10000,
    'evaluation_interval': 1000,
    'max_examples': 20000,
    'delta_easy': 1e-3,
    'delta_hard': 1e-7,
    'seed0': 0,
    'seed1': 100,
    'update_delta_dropped_accuracy': 0.8,
    'num_runs': 2,
    'model': 'UpdatableHoeffdingTreeClassifier',
    'stream_type': 'RandomRBF',
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

# FUNCTIONS
def calculate_average_accuracy(correct_predictions, total_predictions):
    return correct_predictions / total_predictions if total_predictions else 0

def prequential_evaluation(model, stream, config):
    for x, y in stream.take(config['max_examples']):
        prediction = model.predict_one(x)
        model.learn_one(x, y)
        yield prediction, y

def prequential_evaluation_with_queue(model, stream, config, accuracy_event_queue):

    results = []
    for prediction, y in prequential_evaluation(model, stream, config):
        accuracy_event_queue.put((prediction, y)) 
        results.append((prediction, y))
    return results

def data_stream_template_factory(stream_type, preinitialized_params):
    def constructor(seed):
        if stream_type == 'RandomTree':
            return RandomTree(seed_tree=seed, seed_sample=seed, **preinitialized_params)
        elif stream_type == 'RandomRBF':
            return RandomRBF(seed_model=seed, seed_sample=seed, **preinitialized_params)
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    return constructor

def run_experiment(config, seed0, seed1, stream_type):
    # Create queues for this experiment
    accuracy_event_queue = queue.Queue()
    state_update_queue = queue.Queue()

    stream_factory = data_stream_template_factory(stream_type, config['streams'][stream_type])

    concept_drift_stream = ConceptDriftStream(
        stream=stream_factory(seed=seed0), 
        drift_stream=stream_factory(seed=seed1), 
        position=config['change_point'], 
        seed=seed0
    )

    # Use the dictionary to get the class object
    ModelClass = model_classes[config['model']]
    model = ModelClass(delta=config['delta_hard'])

    # Create a state and start the StateUpdater thread
    state = State(config, state_update_queue)
    state_updater = StateUpdater(state, accuracy_event_queue)
    state_updater.start()

    # Create an agent and start the AgentListener thread
    agent = Agent(model, config)
    agent_listener = AgentListener(agent, state_update_queue)
    agent_listener.start()

    # Convert generator to DataFrame before returning
    results = list(prequential_evaluation_with_queue(model, concept_drift_stream, config, accuracy_event_queue))
    results_df = pd.DataFrame(results, columns=['Prediction', 'Actual'])

    # Calculate 'Correct_Classification' column
    results_df['Correct_Classification'] = results_df['Prediction'] == results_df['Actual']

    return results_df

def run_seeded_experiments(config, stream_type):
    seed0, seed1 = config['seed0'], config['seed1']
    seeds0 = range(seed0, seed0 + config['num_runs'])
    seeds1 = range(seed1, seed1 + config['num_runs'])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, [config]*config['num_runs'], seeds0, seeds1, [stream_type]*config['num_runs']))

    # Create an OrderedDict where the keys are the thread numbers and the values are the results
    results_dict = OrderedDict((i, result) for i, result in enumerate(results))

    return results_dict

def run_experiments_multiple_configurations(configurations, default_config):
    all_results = {}

    for config in configurations:
        changed_features = {k: v for k, v in config.items() if v != default_config.get(k)}
        results = {}

        for stream_type in config['streams']:
            results[stream_type] = run_seeded_experiments(config, stream_type)

        all_results[str(changed_features)] = results

    return all_results

# CLASSES
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


class StateUpdater(threading.Thread):
    def __init__(self, state, accuracy_event_queue):
        super().__init__()
        self.state = state
        self.daemon = True  # Ensure thread exits when main program finishes
        self.accuracy_event_queue = accuracy_event_queue

    def run(self):
        while True:
            # Wait for a prediction and true value from the event_queue
            prediction, y = self.accuracy_event_queue.get()

            # Update the state with the prediction and true value
            self.state.update(prediction, y)
            
class State:
    def __init__(self, config, state_update_queue):
        self.config = config

        self.step = 0
        self.evaluation_steps = []

        self.correct_predictions = 0
        self.total_predictions = 0

        self.all_epoch_accuracies = [] 

        self.last_10_accuracies = []
        self.avg_last_10_epoch_accuracies = 0
        self.avg_accuracies = []  # Store average accuracies

        self.state_update_queue = state_update_queue

    def update(self, prediction, y):
        if prediction == y:
            self.correct_predictions += 1
        self.total_predictions += 1
        self.step += 1

        if self.step % self.config['evaluation_interval'] == 0:
            accuracy = self.calculate_average_accuracy()
            self.avg_accuracies.append(accuracy)
            self.all_epoch_accuracies.append(accuracy)  # renamed from accuracies
            self.evaluation_steps.append(self.step)
            self.correct_predictions = 0
            self.total_predictions = 0

            # Update the last 10 accuracies
            self.last_10_accuracies.append(accuracy)
            if len(self.last_10_accuracies) > 10:
                self.last_10_accuracies.pop(0)  # Remove the oldest accuracy

            # Update the average of the last 10 accuracies
            self.avg_last_10_epoch_accuracies = self.get_average_of_last_10_accuracies()

            # Put the updated state into the state_update_queue
            self.state_update_queue.put((self.last_10_accuracies[-1], self.avg_last_10_epoch_accuracies))


    def calculate_average_accuracy(self):
        return self.correct_predictions / self.total_predictions if self.total_predictions else 0

    def get(self):
        data = list(zip(self.evaluation_steps, self.all_epoch_accuracies))  # renamed from accuracies
        df = pd.DataFrame(data, columns=['Evaluation Step', 'Correct_Classification'])
        return df

    def get_last_10_accuracies(self):
        return self.last_10_accuracies

    def get_average_of_last_10_accuracies(self):
        if self.last_10_accuracies:
            return sum(self.last_10_accuracies) / len(self.last_10_accuracies)
        else:
            return 0
        
    def get_avg_accuracies_df(self):
        """Return a DataFrame with average accuracies every 1000 epochs."""
        df = pd.DataFrame(self.avg_accuracies, columns=['Average Accuracy'])
        df.index = df.index * self.config['evaluation_interval']  # Set the index to the epoch number
        return df

class Agent:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def choose_and_apply_action(self, state):
        last_epoch_accuracy, avg_last_10_epoch_accuracy = state
        if last_epoch_accuracy < avg_last_10_epoch_accuracy * self.config['update_delta_dropped_accuracy']:
            self.model.delta = self.config['delta_easy']
        else:
            self.model.delta = self.config['delta_hard']

class AgentListener(threading.Thread):
    def __init__(self, agent, state_update_queue):
        super().__init__()
        self.agent = agent
        self.daemon = True  # Ensure thread exits when main program finishes
        self.state_update_queue = state_update_queue

    def run(self):
        while True:
            # Wait for an state update event
            state = self.state_update_queue.get()

            # Update the agent's state and choose an action
            self.agent.choose_and_apply_action(state)


def main():
    # Set random seeds
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    delta_easy_values = [10**(-i) for i in range(3, 10)]
    update_delta_dropped_accuracy_values = [0.6, 0.8, 0.9, 0.95, 0.99]

    default_config = {
        **CONFIG,
        'delta_easy': 1e-3,
        'delta_hard': 1e-7,
    }

    configurations = [
        {
            **CONFIG,
            'delta_easy': delta_easy,
            'update_delta_dropped_accuracy': update_delta_dropped_accuracy,
        }
        for delta_easy in delta_easy_values
        for update_delta_dropped_accuracy in update_delta_dropped_accuracy_values
    ]

    results = run_experiments_multiple_configurations(configurations, default_config)

    # Now `results` is a dictionary where the keys are the string representations of the changed features and the values are the results of the experiments.

    dfs = {}

    for changed_features, streams in results.items():
        for stream_type, results_dict in streams.items():
            for thread_number, df in results_dict.items():
                # Calculate average epoch-wise accuracy for each run
                avg_accuracy = df['Correct_Classification'].mean()

                # Create a key for the stream-type and thread-number combination
                key = (stream_type, thread_number)

                # If the key is not in the dfs dictionary, add it with an empty list
                if key not in dfs:
                    dfs[key] = []

                # Append the results to the list
                dfs[key].append({'Changed Features': changed_features, 'Average Accuracy': avg_accuracy})

    # Convert the lists to DataFrames using pd.concat
    for key, data in dfs.items():
        dfs[key] = pd.DataFrame(data)

    # Print the DataFrames
    for key, df in dfs.items():
        print(f"Stream Type: {key[0]}, Thread Number: {key[1]}")
        #print(df)
        print(df.to_numpy())
if __name__ == "__main__":    
    main()
