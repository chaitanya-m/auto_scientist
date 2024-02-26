# Import necessary libraries
from river.datasets.synth import RandomRBF, RandomTree, ConceptDriftStream
from river import tree

import pandas as pd
import random
import numpy as np

import concurrent.futures
import queue
import threading

# Create a global queue for events
event_queue = queue.Queue()

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
    'update_delta_accuracy_threshold': 0.98,
    'num_runs': 10,
    'model': 'UpdatableHoeffdingTreeClassifier',
    'stream_type': 'RandomRBF',
    'streams': {
        'RandomRBF': {
            'n_classes': 3,
            'n_features': 2,
            'n_centroids': 3,
        },
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
    state = State()
    agent = Agent(model, config['delta_easy'], config['delta_hard'])

    accuracies = []
    accuracy_changes = []
    step = 0
    correct_predictions = 0
    total_predictions = 0
    prev_accuracy = 0
    epoch_accuracies = []

    for x, y in stream.take(config['max_examples']):
        prediction = model.predict_one(x)
        model.learn_one(x, y)

        if prediction == y:
            correct_predictions += 1
        total_predictions += 1

        step += 1
        if step % config['evaluation_interval'] == 0:
            accuracy = calculate_average_accuracy(correct_predictions, total_predictions)
            state.update(accuracy)

            # Emit an accuracy update event instead of calling the agent directly
            event_queue.put(state.get())  # Emit the entire state

            # Now you can get the state of the environment at any time
            # last_epoch_accuracy, avg_last_10_epoch_accuracy = env.get_state()

            accuracy_change = accuracy - prev_accuracy
            prev_accuracy = accuracy

            accuracies.append(accuracy)
            accuracy_changes.append(accuracy_change)

            epoch_accuracies.append(accuracy)
            if len(epoch_accuracies) > 10:
                epoch_accuracies.pop(0)

            correct_predictions = 0
            total_predictions = 0


    evaluation_steps = list(range(config['evaluation_interval'], config['max_examples'] + 1, config['evaluation_interval']))
    data = list(zip(evaluation_steps, accuracy_changes[0:], accuracies[0:]))
    df = pd.DataFrame(data, columns=['Evaluation Step', 'Change in Accuracy', 'Accuracy'])

    return df

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

    return prequential_evaluation(model, concept_drift_stream, config)

def run_seeded_experiments(config, stream_type):
    seed0, seed1 = config['seed0'], config['seed1']
    seeds0 = range(seed0, seed0 + config['num_runs'])
    seeds1 = range(seed1, seed1 + config['num_runs'])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        dfs = list(executor.map(run_experiment, [config]*config['num_runs'], seeds0, seeds1, [stream_type]*config['num_runs']))

    return dfs

def run_experiments_with_different_policies(config):
    # Run experiments
    for stream_type in config['streams']:
        results = run_seeded_experiments(config, stream_type)

        # Calculate and print the average accuracy after the change point for each pair of experiments
        for i in range(config['num_runs']):
            df = results[i]

            avg_accuracy = df[df['Evaluation Step'] > config['change_point']]['Accuracy'].mean()

            print(f"Experiment {i+1} for stream {stream_type}:")
            print(f"Average accuracy after drift: {avg_accuracy}")
            print("\n")

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

class State:
    def __init__(self):
        self.last_epoch_accuracy = 0
        self.last_10_epoch_accuracies = []

    def update(self, accuracy):
        self.last_epoch_accuracy = accuracy
        self.last_10_epoch_accuracies.append(accuracy)
        if len(self.last_10_epoch_accuracies) > 10:
            self.last_10_epoch_accuracies.pop(0)

    def get(self):
        avg_last_10_epoch_accuracy = sum(self.last_10_epoch_accuracies) / len(self.last_10_epoch_accuracies) if self.last_10_epoch_accuracies else 0
        return self.last_epoch_accuracy, avg_last_10_epoch_accuracy

class Agent:
    def __init__(self, model, delta_easy, delta_hard):
        self.model = model
        self.delta_easy = delta_easy
        self.delta_hard = delta_hard

    def choose_and_apply_action(self, state):
        last_epoch_accuracy, avg_last_10_epoch_accuracy = state
        if last_epoch_accuracy < avg_last_10_epoch_accuracy:
            self.model.delta = self.delta_easy
        else:
            self.model.delta = self.delta_hard

class AgentListener(threading.Thread):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.daemon = True  # Ensure thread exits when main program finishes

    def run(self):
        while True:
            # Wait for an accuracy update event
            accuracy = event_queue.get()

            # Update the agent's state and choose an action
            self.agent.choose_and_apply_action(accuracy)


def main():
    # Set random seeds
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # Create an agent and start the listener thread
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])
    agent = Agent(model, CONFIG['delta_easy'], CONFIG['delta_hard'])
    listener = AgentListener(agent)
    listener.start()

    for stream_type in CONFIG['streams']:
        evaluation_results = run_seeded_experiments(CONFIG, stream_type)

        for i, result in enumerate(evaluation_results):
            print(f"Result {i+1} for stream {stream_type}:")
            print(result)
            print("\n")

    run_experiments_with_different_policies(CONFIG)

if __name__ == "__main__":    
    main()