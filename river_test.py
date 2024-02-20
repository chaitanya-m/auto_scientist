# Import necessary libraries
from river.datasets.synth import RandomRBF, ConceptDriftStream
from river import linear_model, tree
import matplotlib.pyplot as plt
from collections import deque

# Define constants with named attributes
change_point = 3000
evaluation_interval = 1000
max_examples = 5000  # Adjust this value as needed
rolling_window_size = 1000

# Create RandomRBF generators with different seeds
generator1 = RandomRBF(
    seed_model=42, seed_sample=42,
    n_classes=3, n_features=2, n_centroids=3
)
generator2 = RandomRBF(
    seed_model=43, seed_sample=43,
    n_classes=3, n_features=2, n_centroids=3
)

# Create ConceptDriftStream with gradual shift at change_point
drift_stream = ConceptDriftStream(
    stream=generator1, drift_stream=generator2, position=change_point
)

# Define and initialize Perceptron model
model = tree.HoeffdingTreeClassifier()

# Store accuracy values for plotting
accuracies = []
rolling_window = deque(maxlen=rolling_window_size)
rolling_sum = 0.0

# Train and evaluate prequentially
step = 0
correct_predictions = 0
total_predictions = 0
for x, y in drift_stream.take(max_examples):
    
    # Make a prediction
    prediction = model.predict_one(x)

    print(x,y, prediction, prediction==y)

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
print(f"Average accuracy measurements: {accuracies}")
