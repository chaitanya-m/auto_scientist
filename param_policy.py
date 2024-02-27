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

def run_experiment_wrapper(args):
    agent, new_config, seed0, seed1, stream_type = args
    return run_experiment(agent, new_config, seed0, seed1, stream_type)


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



class Agent:
    def __init__(self, config, model, stream_factory, exploration_step_sizes, num_episodes, num_seeds):
        self.config = config
        self.exploration_step_sizes = exploration_step_sizes
        self.num_episodes = num_episodes
        self.num_seeds = num_seeds
        self.baseline = None
        self.model = model
        self.stream_factory = stream_factory

    def find_best_parameters(self):

        # let's write some pseudocode for the agent
        # First create a Q-table. 
        # In the Q table:
        #   Each state is ((last_epoch_average_accuracy - last_10_epoch_average_accuracy) / last_10_epoch_average_accuracy)
        #   Each action is an adjustment for the parameters
        # For each episode
        #   create a set of num_seeds concept drift streams... just increment the seed by 1 for each stream
        #   for all seeds, run in parallel
        #       the experiment without any agent intervention
        #   then for all seeds, run in parallel
        #       the experiment with agent intervention; the agent is allowed to adjust the parameters according to the step sizes
        #   For each seed, if the strategy with agent intervention has a higher reward than the strategy without agent intervention
        #       then update the Q-table with the new reward, which is the difference between the two stream prediction accuracies



        
        for episode in range(self.num_episodes):
            concept_drift_stream = ConceptDriftStream(
                stream=self.stream_factory.create(seed=self.config['seed0']), 
                drift_stream=self.stream_factory.create(seed=self.config['seed1']), 
                position=self.config['change_point'], 
                seed=self.config['seed0']
            )

            action = self.choose_action()
            reward = experiment.run(action)
            if self.baseline is None:
                self.baseline = reward
            if reward > self.baseline:
                self.baseline = reward
            if self.baseline - reward > self.drop_threshold:
                self.adjust_parameters(experiment)
        return self.config

    def choose_action(self):
        action = {param: np.random.choice([-step['coarse'], -step['fine'], step['fine'], step['coarse']])
                  for param, step in self.exploration_step_sizes.items()}
        return action

    def adjust_parameters(self, experiment):
        params = list(self.config.keys())
        increase_param = random.choice(params)
        decrease_param = random.choice([param for param in params if param != increase_param])
        self.config[increase_param] += self.exploration_step_sizes[increase_param]['fine']
        self.config[decrease_param] -= self.exploration_step_sizes[decrease_param]['fine']
        self.config = experiment.ensure_bounds(self.config)

def main():

    # Setup stream factory
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])

    # Setup model
    # Use the dictionary to get the class object
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])

    # Setup and run Agent
    exploration_step_sizes = {
        'delta_easy': {'coarse': 100, 'fine': 10},
        'update_delta_dropped_accuracy': {'coarse': 0.1, 'fine': 0.01},
    }
    agent = Agent(CONFIG, model, stream_factory, exploration_step_sizes, num_episodes=10, num_seeds=5)
    best_parameters = agent.find_best_parameters()

    # Print the best parameters found
    print(f"Best parameters found: {best_parameters}")

if __name__ == "__main__":
    main()








