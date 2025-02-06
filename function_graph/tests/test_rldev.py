import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # Force CPU usage.
os.environ["TF_DETERMINISTIC_OPS"] = "1"          # Request deterministic operations.

import random
import numpy as np
import tensorflow as tf
import unittest

# Seed all relevant random generators.
random.seed(1)
np.random.seed(42)
tf.random.set_seed(42)

import keras
from keras import layers, initializers
from graph.node import SubGraphNode
from utils.rl_dev import RLEnvironment, DummyAgent, run_episode, create_minimal_network


def visualize_graph(composer):
    """
    Prints a visual summary of the graph constructed by the composer.
    For each node, it prints:
      - The node name and its type.
      - For InputNodes: the input shape.
      - For SingleNeuron nodes: the activation function and the weights (if the model is built).
      - For SubGraphNodes: a summary of the contained Keras model and the weights of each layer.
    Finally, it prints the connection list.
    """
    print("=== Graph Visualizer ===")
    print("Nodes:")
    for node_name, node in composer.nodes.items():
        print(f"Node: {node_name} ({node.__class__.__name__})")
        # If the node has an input shape attribute (e.g. InputNode), print it.
        if hasattr(node, "input_shape"):
            print(f"  Input shape: {node.input_shape}")
        # For SingleNeuron, print activation and (if available) its weights.
        if node.__class__.__name__ == "SingleNeuron":
            print(f"  Activation: {node.activation}")
            try:
                # Assuming the model has been built, retrieve the corresponding layer.
                layer = composer.keras_model.get_layer(node_name)
                weights = layer.get_weights()
                print(f"  Weights: {weights}")
            except Exception as e:
                print("  Weights not available (model not built yet).")
        # For SubGraphNode, print the internal Keras model summary and each layer's weights.
        if node.__class__.__name__ == "SubGraphNode":
            print("  Contains a Keras model:")
            node.model.summary()
            for layer in node.model.layers:
                print(f"    Layer {layer.name} weights: {layer.get_weights()}")
    print("\nConnections:")
    for target, connections in composer.connections.items():
        print(f"Node '{target}' has incoming connections:")
        for parent, merge_mode in connections:
            print(f"  From: '{parent}', merge mode: {merge_mode}")
    print("=== End of Graph ===")



def train_learned_abstraction_model(env, epochs=1000):
    """
    Helper function to build and train a learned abstraction model.
    This model consists of:
      - an input layer (matching env.features, i.e. 2 features),
      - a hidden dense layer with 3 neurons (ReLU activation),
      - an output dense layer with 1 neuron and sigmoid activation.
    It trains the full model on env.features and env.true_labels,
    prints the full-model accuracy, then extracts the hidden layer model and wraps it in a SubGraphNode.
    """
    kernel_init = initializers.GlorotUniform(seed=42)
    input_shape = (2,)
    new_input = layers.Input(shape=input_shape, name="sub_input")
    hidden = layers.Dense(3, activation='relu', name="hidden_layer", kernel_initializer=kernel_init)(new_input)
    output = layers.Dense(1, activation='sigmoid', name="output_layer", kernel_initializer=kernel_init)(hidden)
    full_model = keras.models.Model(new_input, output, name="learned_abstraction_model_full")
    full_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    full_model.fit(env.features, env.true_labels, epochs=epochs, verbose=0)
    loss, acc = full_model.evaluate(env.features, env.true_labels, verbose=0)
    print(f"Learned abstraction full model accuracy after {epochs} epochs: {acc:.3f}")
    
    # Create a model that outputs the hidden layer.
    abstraction_model = keras.models.Model(new_input, hidden, name="learned_abstraction_model_extracted")
    subgraph_node = SubGraphNode(name="learned_abstraction", model=abstraction_model)
    return subgraph_node

class VerboseTestCase(unittest.TestCase):
    def assertGreaterVerbose(self, first, second, msg=None):
        """Assert that first > second and print the provided message with a checkmark if it passes."""
        # Call the built-in assertion. This will raise an error if the condition fails.
        self.assertGreater(first, second, msg)
        # If we get here, the assertion passed.
        success_msg = msg if msg else f"assertGreater passed: {first} > {second}"
        print(f"✓ {success_msg}")

    def assertTrueVerbose(self, expr, msg=None):
        """Assert that expr is True and print the provided message with a checkmark if it passes."""
        self.assertTrue(expr, msg)
        success_msg = msg if msg else "assertTrue passed"
        print(f"✓ {success_msg}")

def assert_allclose_verbose(a, b, atol=1e-7, err_msg=""):
    try:
        np.testing.assert_allclose(a, b, atol=atol, err_msg=err_msg)
        print(f"✓ {err_msg if err_msg else 'assert_allclose passed'}")
    except AssertionError as e:
        raise e

class TestLearnedAbstractionTraining(VerboseTestCase):
    def test_learned_abstraction_training(self):
        """
        This test:
          1. Creates a minimal network with one input node and one output node (with sigmoid activation).
          2. Uses a helper to train a learned abstraction (a hidden layer with 3 neurons) on the fixed dataset.
          3. Adds the learned abstraction into the network by connecting input -> abstraction -> output.
          4. Freezes the learned abstraction so its weights remain unchanged.
          5. Saves the learned abstraction's weights.
          6. Rebuilds the Keras model via the composer.
          7. Fine-tunes the composed model for 500 epochs (only the output neuron weights should change).
          8. Evaluates and prints the final accuracy.
          9. Verifies that the learned abstraction's weights are identical before and after fine-tuning.
          10. Stores the learned abstraction in the repository.
          11. Asserts that the final accuracy is above a chosen threshold.
        """
        # Create the environment.
        env = RLEnvironment(total_steps=1, num_instances_per_step=100, seed=0)
        features, true_labels = env.features, env.true_labels

        # Build the minimal network.
        composer, model = create_minimal_network(input_shape=(2,))
        
        # Train the learned abstraction.
        learned_abstraction = train_learned_abstraction_model(env, epochs=1000)

        # Freeze the learned abstraction so its weights remain unchanged during fine-tuning.
        learned_abstraction.model.trainable = False
        # Save weights before fine-tuning.
        weights_before = learned_abstraction.model.get_weights()

        # Add the learned abstraction into the network.
        composer.add_node(learned_abstraction)
        composer.connect("input", learned_abstraction.name)
        composer.connect(learned_abstraction.name, "output")
        
        # Rebuild the Keras model from the updated composer.
        model = composer.build()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        visualize_graph(composer)

        # Fine-tune the composed model; only the output neuron's weights should be updated.
        model.fit(features, true_labels, epochs=300, verbose=0)

        visualize_graph(composer)


        # Evaluate the fine-tuned model.
        loss, acc = model.evaluate(features, true_labels, verbose=0)
        print(f"Trained network accuracy after integrating learned abstraction and fine-tuning: {acc:.3f}")
        
        # Save weights after fine-tuning.
        weights_after = learned_abstraction.model.get_weights()

        # Assert that the learned abstraction's weights have not changed.
        for w_before, w_after in zip(weights_before, weights_after):
            assert_allclose_verbose(w_before, w_after, atol=1e-7,
                                       err_msg="Learned abstraction weights should remain unchanged after fine-tuning.")
        
        # Store the learned abstraction in the repository.
        env.repository["learned_abstraction"] = learned_abstraction
        
        # Assert that the final accuracy is above a chosen threshold (e.g., 0.9).
        self.assertGreaterVerbose(acc, 0.9, "Trained network accuracy should be above 0.9.")

# class TestReuseAdvantage(unittest.TestCase):
#     def test_reuse_advantage(self):
#         """
#         Hypothesis: When one agent adds the learned abstraction and the other does not,
#         the agent that adds it will obtain higher accuracy and receive a reward advantage.
        
#         Test: Pre-populate the repository with the learned abstraction.
#         Run an episode where agent 0 chooses "add_abstraction" on the first step,
#         while agent 1 always chooses "no_change". Print detailed step-by-step accuracies and rewards,
#         and verify that agent 0's reward on the first step is higher than agent 1's.
#         """
#         env = RLEnvironment(total_steps=5, num_instances_per_step=100, seed=0)
#         learned_abstraction = train_learned_abstraction_model(env, epochs=1000)
#         env.repository["learned_abstraction"] = learned_abstraction

#         # Define action plans.
#         action_plan0 = ["add_abstraction"] + ["no_change"] * 4
#         action_plan1 = ["no_change"] * 5

#         agent0 = DummyAgent(action_plan={0: action_plan0})
#         agent1 = DummyAgent(action_plan={1: action_plan1})
        
#         # run_episode returns three objects: actions_history, rewards_history, accuracies_history.
#         actions, rewards, accuracies = run_episode(env, agent0, agent1)
        
#         print("\nDetailed Step-by-Step Output:")
#         for step in range(len(accuracies[0])):
#             print(f"Step {step+1}:")
#             print(f"  Agent 0: Accuracy = {accuracies[0][step]:.3f}, Reward = {rewards[0][step]:.3f}")
#             print(f"  Agent 1: Accuracy = {accuracies[1][step]:.3f}, Reward = {rewards[1][step]:.3f}")

#         reward0 = rewards[0][0]
#         reward1 = rewards[1][0]
#         diff = reward0 - reward1
#         print(f"\nTest outcome: Agent 0's reward on step 1 = {reward0}, Agent 1's reward = {reward1}.")
#         print(f"Agent 0 won by a margin of {diff} reward points on the first step.")
        
#         self.assertGreater(reward0, reward1,
#                            "Agent 0 should receive a higher reward than Agent 1 when using the learned abstraction.")

if __name__ == '__main__':
    unittest.main()
