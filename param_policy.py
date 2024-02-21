# Import necessary libraries
from river.datasets.synth import RandomRBF, ConceptDriftStream
from river import linear_model, tree
import matplotlib.pyplot as plt
from collections import deque

import random
import numpy as np
import pandas as pd
random.seed(0)
np.random.seed(0)

# CONSTANTS

# Define constants with named attributes
change_point = 10000
evaluation_interval = 1000
max_examples = 20000  # Adjust this value as needed

delta_easy = 1e-3
delta_hard = 1e-7

seed1 = 0
seed2 = 3


# Create RandomRBF generators with different seeds
generator1 = RandomRBF(
    seed_model=seed1, seed_sample=seed1,
    n_classes=3, n_features=2, n_centroids=3
)
generator2 = RandomRBF(
    seed_model=seed2, seed_sample=seed2,
    n_classes=3, n_features=2, n_centroids=3
)

# Create ConceptDriftStream with gradual shift at change_point
drift_stream = ConceptDriftStream(
    stream=generator1, drift_stream=generator2, position=change_point, seed=seed1
)

# FUNCTIONS

# Prequential training and evaluation function
import pandas as pd

def prequential_evaluation(model=None, stream=None, max_examples=20000, 
                           evaluation_interval=1000, change_point=10000, 
                           delta_easy = 1e-3, delta_hard=1e-3, 
                           update_delta_when_accuracy_drops=False, 
                           update_delta_accuracy_threshold = 0.8):
    accuracies = []
    accuracy_changes = []

    step = 0
    correct_predictions = 0
    total_predictions = 0
    prev_accuracy = 0
    for x, y in stream.take(max_examples):
        prediction = model.predict_one(x)
        model.learn_one(x, y)

        if prediction == y:
            correct_predictions += 1
        total_predictions += 1

        step += 1
        if step % evaluation_interval == 0:
            accuracy = correct_predictions / total_predictions
            accuracy_change = accuracy - prev_accuracy
            prev_accuracy = accuracy

            accuracies.append(accuracy)
            accuracy_changes.append(accuracy_change)

            correct_predictions = 0
            total_predictions = 0

            if update_delta_when_accuracy_drops:
                if accuracy < update_delta_accuracy_threshold:
                    model.update_delta(delta_easy)
                else:
                    model.update_delta(delta_hard)

    evaluation_steps = list(range(evaluation_interval, max_examples + 1, evaluation_interval))
    data = list(zip(evaluation_steps, accuracy_changes[1:], accuracies[1:]))
    df = pd.DataFrame(data, columns=['Evaluation Step', 'Change in Accuracy', 'Accuracy'])

    return df





# CLASSES

class UpdatableHoeffdingTreeClassifier(tree.HoeffdingTreeClassifier):
    def __init__(self, delta=delta_hard):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta



# MAIN
        
# Create an instance of your custom class
model = UpdatableHoeffdingTreeClassifier()

# Run the prequential evaluation function
df = prequential_evaluation(model, drift_stream, max_examples, evaluation_interval, change_point, delta_easy, delta_hard, update_delta_when_accuracy_drops=True)

print(df)

