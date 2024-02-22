# Import necessary libraries
from river.datasets.synth import RandomRBF, ConceptDriftStream
from river import tree
import pandas as pd
import random
import numpy as np


# CONSTANTS
CONFIG = {
    'random_seed': 0,
    'num_runs': 10,
    'model': 'UpdatableEFDTClassifier',
    'update_delta_when_accuracy_drops': True,
    'update_delta_accuracy_threshold': 0.9,

    'stream': {
        'type': 'RandomRBF',
        'change_point': 10000,
        'max_examples': 20000,
        'evaluation_interval': 1000,
        'seeds': {
            'initial': 0,
            'drift': 100,
        },
        'params': {
            'n_classes': 5,
            'n_features': 3,
            'n_centroids': 5,
        },
    },

    'model_params': {
        'delta_easy': 4e-1,
        'delta_hard': 1e-7,
    },
}

# Set random seeds
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# FUNCTIONS
def prequential_evaluation(model, stream, config):
    accuracies = []
    accuracy_changes = []
    step = 0
    correct_predictions = 0
    total_predictions = 0
    prev_accuracy = 0
    for x, y in stream.take(config['stream']['max_examples']):
        prediction = model.predict_one(x)
        model.learn_one(x, y)

        if prediction == y:
            correct_predictions += 1
        total_predictions += 1

        step += 1
        if step % config['stream']['evaluation_interval'] == 0:
            accuracy = correct_predictions / total_predictions
            accuracy_change = accuracy - prev_accuracy
            prev_accuracy = accuracy

            accuracies.append(accuracy)
            accuracy_changes.append(accuracy_change)

            correct_predictions = 0
            total_predictions = 0

            if config['update_delta_when_accuracy_drops']:
                if 'update_delta_accuracy_threshold' in config['model_params']:
                    if accuracy < config['model_params']['update_delta_accuracy_threshold']:
                        model.update_delta(config['model_params']['delta_easy'])
                    else:
                        model.update_delta(config['model_params']['delta_hard'])

    evaluation_steps = list(range(config['stream']['evaluation_interval'], config['stream']['max_examples'] + 1, config['stream']['evaluation_interval']))
    data = list(zip(evaluation_steps, accuracy_changes[0:], accuracies[0:]))
    df = pd.DataFrame(data, columns=['Evaluation Step', 'Change in Accuracy', 'Accuracy'])

    return df

def data_stream_template_factory(stream_type, preinitialized_params):
    def constructor(seed):
        if stream_type == 'RandomRBF':
            return RandomRBF(seed_model=seed, seed_sample=seed, **preinitialized_params)
        elif stream_type == 'RandomTree':
            return synth.RandomTreeGenerator(seed=seed, **preinitialized_params)
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    return constructor
import concurrent.futures

import concurrent.futures

def run_experiment(config, seed0, seed1):
    stream_factory = data_stream_template_factory(config['stream']['type'], config['stream']['params'])

    concept_drift_stream = ConceptDriftStream(
        stream=stream_factory(seed=seed0), 
        drift_stream=stream_factory(seed=seed1), 
        position=config['stream']['change_point'], 
        seed=seed0
    )

    ModelClass = globals()[config['model']]
    model = ModelClass(delta=config['model_params']['delta_hard'])

    return prequential_evaluation(model, concept_drift_stream, config)

def run_seeded_experiments(config):
    dfs = []
    seed0, seed1 = config['stream']['seeds']['initial'], config['stream']['seeds']['drift']
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, config, seed0 + i, seed1 + i) for i in range(config['num_runs'])]
        for future in futures:
            dfs.append(future.result())

    return dfs

def run_experiments_with_different_policies(config):
    # Copy the configuration to avoid side effects
    config_false = config.copy()
    config_true = config.copy()

    # Set 'update_delta_when_accuracy_drops' to False and True respectively
    config_false['update_delta_when_accuracy_drops'] = False
    config_true['update_delta_when_accuracy_drops'] = True

    # Run experiments
    results_false = run_seeded_experiments(config_false)
    results_true = run_seeded_experiments(config_true)

    # Calculate and print the average accuracy after the change point for each pair of experiments
    for i in range(config['num_runs']):
        df_false = results_false[i]
        df_true = results_true[i]

        avg_accuracy_false = df_false[df_false['Evaluation Step'] > config['stream']['change_point']]['Accuracy'].mean()
        avg_accuracy_true = df_true[df_true['Evaluation Step'] > config['stream']['change_point']]['Accuracy'].mean()

        print(f"Experiment {i+1}:")
        print(f"Average accuracy after drift with 'update_delta_when_accuracy_drops' set to False: {avg_accuracy_false}")
        print(f"Average accuracy after drift with 'update_delta_when_accuracy_drops' set to True: {avg_accuracy_true}")
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

# MAIN
def main():
    # evaluation_results = run_seeded_experiments(CONFIG)

    # for i, result in enumerate(evaluation_results):
    #     print(f"Result {i+1}:")
    #     print(result)
    #     print("\n")

    run_experiments_with_different_policies(CONFIG)

if __name__ == "__main__":
    main()