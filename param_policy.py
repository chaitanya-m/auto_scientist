# Import necessary libraries
from river.datasets.synth import RandomRBF, RandomTree, ConceptDriftStream
import concurrent.futures
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
    'seed1': 3,
    'update_delta_accuracy_threshold': 0.98,
    'num_runs': 2,
    'model': 'UpdatableHoeffdingTreeClassifier',
    'stream_type': 'RandomRBF',
    'update_delta_when_accuracy_drops': True,
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

def check_for_drift(epoch_accuracies, config, model):
    if len(epoch_accuracies) == 10:
        avg_accuracy_last_epoch = epoch_accuracies[-1]
        avg_accuracy_last_10_epochs = sum(epoch_accuracies) / len(epoch_accuracies)

        if avg_accuracy_last_epoch < config['update_delta_accuracy_threshold'] * avg_accuracy_last_10_epochs:
            if config['update_delta_when_accuracy_drops']:
                model.update_delta(config['delta_easy'])
        else:
            model.update_delta(config['delta_hard'])

def prequential_evaluation(model, stream, config):
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
            accuracy_change = accuracy - prev_accuracy
            prev_accuracy = accuracy

            accuracies.append(accuracy)
            accuracy_changes.append(accuracy_change)

            epoch_accuracies.append(accuracy)
            if len(epoch_accuracies) > 10:
                epoch_accuracies.pop(0)

            correct_predictions = 0
            total_predictions = 0

            check_for_drift(epoch_accuracies, config, model)

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

    ModelClass = globals()[config['model']]
    model = ModelClass(delta=config['delta_hard'])

    return prequential_evaluation(model, concept_drift_stream, config)

def run_seeded_experiments(config, stream_type):
    dfs = []
    stream_factory = data_stream_template_factory(stream_type, config['streams'][stream_type])
    seed0, seed1 = config['seed0'], config['seed1']

    for i in range(config['num_runs']):
        concept_drift_stream = ConceptDriftStream(
            stream=stream_factory(seed=seed0), 
            drift_stream=stream_factory(seed=seed1), 
            position=config['change_point'], 
            seed=seed0
        )

        ModelClass = globals()[config['model']]
        model = ModelClass(delta=config['delta_hard'])

        dfs.append(prequential_evaluation(model, concept_drift_stream, config))
        
        seed0 += 1
        seed1 += 1
    return dfs

def run_experiments_with_different_policies(config):
    # Copy the configuration to avoid side effects
    config_false = config.copy()
    config_true = config.copy()

    # Set 'update_delta_when_accuracy_drops' to False and True respectively
    config_false['update_delta_when_accuracy_drops'] = False
    config_true['update_delta_when_accuracy_drops'] = True

    # Run experiments
    for stream_type in config['streams']:
        results_false = run_seeded_experiments(config_false, stream_type)
        results_true = run_seeded_experiments(config_true, stream_type)

        # Calculate and print the average accuracy after the change point for each pair of experiments
        for i in range(config['num_runs']):
            df_false = results_false[i]
            df_true = results_true[i]

            avg_accuracy_false = df_false[df_false['Evaluation Step'] > config['change_point']]['Accuracy'].mean()
            avg_accuracy_true = df_true[df_true['Evaluation Step'] > config['change_point']]['Accuracy'].mean()

            print(f"Experiment {i+1} for stream {stream_type}:")
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

    # Set random seeds
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    for stream_type in CONFIG['streams']:
        evaluation_results = run_seeded_experiments(CONFIG, stream_type)

        for i, result in enumerate(evaluation_results):
            print(f"Result {i+1} for stream {stream_type}:")
            print(result)
            print("\n")

    run_experiments_with_different_policies(CONFIG)

if __name__ == "__main__":    
    main()

