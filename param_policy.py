# Import necessary libraries
from river.datasets.synth import RandomRBF, ConceptDriftStream
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
    'update_delta_accuracy_threshold': 0.8,
    'num_runs': 2,
    'stream_type': 'RandomRBF',
    'preinitialized_params_random_rbf': {
        'n_classes': 3,
        'n_features': 2,
        'n_centroids': 3,
    },
    'update_delta_when_accuracy_drops': True,
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
    for x, y in stream.take(config['max_examples']):
        prediction = model.predict_one(x)
        model.learn_one(x, y)

        if prediction == y:
            correct_predictions += 1
        total_predictions += 1

        step += 1
        if step % config['evaluation_interval'] == 0:
            accuracy = correct_predictions / total_predictions
            accuracy_change = accuracy - prev_accuracy
            prev_accuracy = accuracy

            accuracies.append(accuracy)
            accuracy_changes.append(accuracy_change)

            correct_predictions = 0
            total_predictions = 0

            if config['update_delta_when_accuracy_drops']:
                if accuracy < config['update_delta_accuracy_threshold']:
                    model.update_delta(config['delta_easy'])
                else:
                    model.update_delta(config['delta_hard'])

    evaluation_steps = list(range(config['evaluation_interval'], config['max_examples'] + 1, config['evaluation_interval']))
    data = list(zip(evaluation_steps, accuracy_changes[0:], accuracies[0:]))
    df = pd.DataFrame(data, columns=['Evaluation Step', 'Change in Accuracy', 'Accuracy'])

    return df

def data_stream_template_factory(stream_type, preinitialized_params):
    def constructor(seed):
        if stream_type == 'RandomRBF':
            return RandomRBF(seed_model=seed, seed_sample=seed, **preinitialized_params)
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    return constructor

def run_seeded_experiments(config):
    dfs = []
    stream_factory = data_stream_template_factory(config['stream_type'], config['preinitialized_params_random_rbf'])
    seed0, seed1 = config['seed0'], config['seed1']

    for i in range(config['num_runs']):
        concept_drift_stream = ConceptDriftStream(
            stream=stream_factory(seed=seed0), 
            drift_stream=stream_factory(seed=seed1), 
            position=config['change_point'], 
            seed=seed0
        )

        model = UpdatableHoeffdingTreeClassifier(delta=config['delta_hard'])
        dfs.append(prequential_evaluation(model, concept_drift_stream, config))
        
        seed0 += 1
        seed1 += 1
    return dfs

# CLASSES
class UpdatableHoeffdingTreeClassifier(tree.HoeffdingTreeClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta

# MAIN
def main():
    evaluation_results = run_seeded_experiments(CONFIG)

    for i, result in enumerate(evaluation_results):
        print(f"Result {i+1}:")
        print(result)
        print("\n")

if __name__ == "__main__":
    main()