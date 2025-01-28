import data_gen.categorical_classification as categorical_classification
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import random
import os

# Setting up random seed values for reproducibility
def set_random_seeds(seed_value=0):
    # Python random seed
    random.seed(seed_value)
    # NumPy random seed
    np.random.seed(seed_value)
    # TensorFlow random seed
    tf.random.set_seed(seed_value)
    # If using CUDA for GPU support, ensure the random seed is set for deterministic results
    os.environ['PYTHONHASHSEED'] = str(seed_value)


class SimpleNet:

    def __init__(self, input_dim, num_classes, hidden_size=5, hidden_layers=2):
        '''Create a simple neural net for comparison'''
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))  # Input layer
        for _ in range(self.hidden_layers):
            model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Single output for binary classification
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))  # Multiple outputs for multi-class classification
        return model

    def compile(self, optimizer='adam', loss=None):
        if loss is None:
            loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, X, y, epochs=10, verbose=0):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def evaluate(self, X, y, verbose=0):
        _, accuracy = self.model.evaluate(X, y, verbose=verbose)
        return accuracy

    def predict(self, X):
        return self.model.predict(X)


# Data preparation with one-hot encoding for multi-class classification

def prepare_data(dataset, num_classes):
    X = dataset.iloc[:, :-1].values.astype(float)
    y = dataset.iloc[:, -1].values.astype(int)

    # One-hot encode y for multi-class classification if necessary
    if num_classes > 2:
        encoder = OneHotEncoder(sparse_output=False, categories='auto')  # Corrected to sparse_output=False
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    return X, y

# Train and evaluate the neural network
def train_and_evaluate_neural_net(X, y, num_classes):
    input_dim = X.shape[1]
    net = SimpleNet(input_dim, num_classes)

    # Choose appropriate loss function based on number of classes
    if num_classes == 2:
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss_function = 'categorical_crossentropy'

    net.compile(loss=loss_function)

    net.fit(X, y, verbose=0)

    if num_classes == 2:
        # Binary classification: apply sigmoid and compute accuracy
        y_pred_probs = net.predict(X)
        y_pred_class = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to class predictions
        accuracy = np.mean(y == y_pred_class)
    else:
        # Multi-class classification
        accuracy = net.evaluate(X, y, verbose=0)

    return accuracy


# Set random seed for reproducibility
set_random_seeds(42)

# Create an instance of the DataSchemaFactory to generate schemas
factory = categorical_classification.DataSchemaFactory()

# Define the types of schemas to generate, with their parameters
schema_types = {
    "type1": {"num_features": 2, "num_categories": 2, "num_classes": 2, "flatness": 10},
    "type2": {"num_features": 3, "num_categories": 3, "num_classes": 3, "flatness": 5},
}

# Specify the number of schemas to generate for each type
num_schemas_per_type = 10

# Set the starting value for the random seed used in schema generation
random_seed_start = 0

# Create a generator object that will yield schemas and datasets one by one
dataset_generator = factory.generate_schemas_and_datasets(schema_types, num_schemas_per_type, random_seed_start)

# Loop through generated datasets, process them, and run experiments
for i, (schema_type, schema, dataset) in enumerate(dataset_generator):
    print(f"\nProcessing schema type: {schema_type}, Iteration: {i}")
    print(f"Schema details:")
    schema.print_schema()

    print(f"Dataset head:")
    print(dataset.head())

    # 1. Prepare data
    X, y = prepare_data(dataset, schema.num_classes)

    # 2. Train and evaluate
    accuracy = train_and_evaluate_neural_net(X, y, schema.num_classes)

    print(f"Neural network accuracy on dataset: {accuracy}")
