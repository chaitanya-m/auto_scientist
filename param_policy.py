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
def prequential_evaluation(model=None, stream=None, max_examples=20000, evaluation_interval=1000, change_point=10000, delta_easy = 1e-3, delta_hard=1e-3, update_delta_when_accuracy_drops=False, update_delta_accuracy_threshold = 0.0):
    # Store accuracy values for plotting
    accuracies = []

    # Train and evaluate prequentially
    step = 0
    correct_predictions = 0
    total_predictions = 0
    for x, y in stream.take(max_examples):

        # Make a prediction
        prediction = model.predict_one(x)

        # Update the model
        model.learn_one(x, y)

        # Update accuracy count
        if prediction == y:
            correct_predictions += 1
        total_predictions += 1

        # Calculate and store accuracy every evaluation interval
        step += 1
        if step % evaluation_interval == 0:
            accuracy = correct_predictions / total_predictions

            accuracies.append((step, accuracy))

            # Reset counts for the next evaluation interval
            correct_predictions = 0
            total_predictions = 0

            if update_delta_when_accuracy_drops:
                if accuracy < update_delta_accuracy_threshold:
                    # Update delta value
                    model.update_delta(delta_easy)
                else:
                    model.update_delta(delta_hard)

    return accuracies




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
accuracies = prequential_evaluation(model, drift_stream, max_examples, evaluation_interval, change_point, delta_easy, delta_hard, update_delta_when_accuracy_drops=True)


# Extract steps and accuracies from the list
steps = [a[0] for a in accuracies]
accuracies = [a[1] for a in accuracies]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, accuracies, label="Average Accuracy")
plt.xlabel("Time Steps (x 1000)")
plt.ylabel("Accuracy")
plt.title("Prequential Average Accuracy over Time (Window Size = 1000)")
plt.axvline(x=change_point, color="r", linestyle="--", label="Concept Drift")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print accuracy measurements (optional)
# print(f"Average accuracy measurements: {accuracies}")

# Print the timestep:accuracy pairs
# print(f"Step:Accuracy pairs: {list(zip(steps, accuracies))}")

# Print the [evaluation step : change in accuracy between each pair of evaluations] pairs, 
# with accuracies restricted to 3 decimal places, and steps divided by evaluation_interval
print(f"Step:Change in Accuracy pairs: {list(zip([int(s/evaluation_interval) for s in steps[1:]], [f'{a:.3f}' for a in np.diff(accuracies)]))}")

# Print the [evaluation step : accuracy between each pair of evaluations] pairs, 
# with accuracies restricted to 3 decimal places, and steps divided by evaluation_interval
print(f"Step:Accuracy pairs: {list(zip([int(s/evaluation_interval) for s in steps[1:]], [f'{a:.3f}' for a in accuracies[1:]]))}")

# Create a table with the following columns:
# Evaluation step, Change in accuracy between each pair of evaluations, Accuracy between each pair of evaluations
# with accuracies restricted to 3 decimal places, and steps divided by evaluation_interval

df = pd.DataFrame(list(zip([int(s/evaluation_interval) for s in steps[1:]], [f'{a:.3f}' for a in np.diff(accuracies)], [f'{a:.3f}' for a in accuracies[1:]])), columns=['Evaluation Epoch', 'Change in accuracy', 'Accuracy'])

print(df)

