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
from keras import layers, initializers, utils
from graph.node import SubGraphNode
from utils.rl_dev import RLEnvironment, DummyAgent, run_episode, create_minimal_network
from graphviz import Digraph

import os
from graphviz import Digraph
from keras import utils
from graph.node import SubGraphNode

def visualize_graph(composer, output_file_base="graph_view", image_dir="graph_images"):
    """
    Visualizes the composed graph as a single image with subgraph images displayed
    next to their corresponding nodes.

    For each node in the composer:
      - A text-only node is drawn with the node's name and type.
      - If the node is a SubGraphNode, an SVG image of its internal Keras model is generated
        (via keras.utils.plot_model), and a separate image node is added to the graph.
    An invisible edge is created from the original subgraph node to its image node,
    and a subgraph cluster is defined to force them onto the same rank.
    
    Each call increments a counter so that output filenames are unique.
    The final composite graph is rendered as an SVG, automatically opened, and its DOT source is printed.
    """
    # Create directory for images if it does not exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Ensure unique file names by maintaining a counter.
    if not hasattr(visualize_graph, "counter"):
        visualize_graph.counter = 0
    visualize_graph.counter += 1
    output_file = f"{output_file_base}_{visualize_graph.counter}"
    
    dot = Digraph(comment="Neural Net Graph", format="svg")
    
    # Set general graph attributes for layout and font.
    dot.attr(rankdir="LR")
    dot.attr(fontname="Arial", fontsize="10")
    dot.attr('node', fontname="Arial", fontsize="10", shape="box", style="filled", fillcolor="white")
    
    # Dictionary to store names of image nodes for each SubGraphNode.
    subgraph_image_nodes = {}
    
    # Add each node in the composer.
    for node_name, node in composer.nodes.items():
        label = f"{node_name}\\n({node.__class__.__name__})"
        dot.node(node_name, label=label)
        if isinstance(node, SubGraphNode):
            image_filename = os.path.join(image_dir, f"{node_name}_{visualize_graph.counter}.svg")
            try:
                # Generate an SVG image of the subgraph's internal Keras model.
                utils.plot_model(node.model, to_file=image_filename, show_shapes=True, dpi=60)
                # Create a separate node for the image.
                image_node_name = f"{node_name}_img"
                dot.node(image_node_name, label="", image=image_filename, shape="none")
                # Add an invisible edge to force the image node to appear next to the original node.
                dot.edge(node_name, image_node_name, style="dashed", constraint="false")
                subgraph_image_nodes[node_name] = image_node_name
            except Exception as e:
                print(f"Error generating plot for node '{node_name}': {e}")
    
    # Add the edges from the composer.
    for target, connections in composer.connections.items():
        for parent, merge_mode in connections:
            dot.edge(parent, target, label=merge_mode)
    
    # Create subgraph clusters for each SubGraphNode and its image node so that they appear in the same rank.
    for node_name, image_node in subgraph_image_nodes.items():
        with dot.subgraph() as s:
            s.attr(rank="same")
            s.node(node_name)
            s.node(image_node)
    
    # Render the graph, save to file, and open it.
    dot.render(output_file, view=True)
    print(f"Graph visualized and saved to {output_file}.svg")
    
    # Print the DOT source for debugging.
    print("\nGraph DOT source:")
    print(dot.source)


def print_graph_nodes(composer):
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
    It trains the full model on 500 instances generated from the environment's schema,
    prints the full-model accuracy, then extracts the hidden layer model and wraps it in a SubGraphNode.
    """
    from keras import layers, initializers, models
    kernel_init = initializers.GlorotUniform(seed=42)
    input_shape = (2,)
    new_input = layers.Input(shape=input_shape, name="sub_input")
    hidden = layers.Dense(3, activation='relu', name="hidden_layer", kernel_initializer=kernel_init)(new_input)
    output = layers.Dense(1, activation='sigmoid', name="output_layer", kernel_initializer=kernel_init)(hidden)
    full_model = models.Model(new_input, output, name="learned_abstraction_model_full")
    full_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Generate a new dataset of 500 examples from the fixed schema.
    df = env.schema.generate_dataset(num_instances=500)
    features = df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
    true_labels = df["label"].to_numpy(dtype=int)
    
    full_model.fit(features, true_labels, epochs=epochs, verbose=0)
    loss, acc = full_model.evaluate(features, true_labels, verbose=0)
    print(f"Learned abstraction full model accuracy after {epochs} epochs on 500 instances: {acc:.3f}")
    
    # Create a model that outputs the hidden layer.
    abstraction_model = models.Model(new_input, hidden, name="learned_abstraction_model_extracted")
    from graph.node import SubGraphNode
    subgraph_node = SubGraphNode(name="learned_abstraction", model=abstraction_model)
    return subgraph_node


class TestLearnedAbstractionTraining(unittest.TestCase):
    def test_learned_abstraction_training(self):
        """
        This test:
          1. Creates a minimal network with one input node and one output node (with sigmoid activation).
          2. Uses a helper to train a learned abstraction (a hidden layer with 3 neurons) on a dataset of 500 examples.
          3. Adds the learned abstraction into the network by connecting input -> abstraction -> output.
          4. Freezes the learned abstraction so its weights remain unchanged.
          5. Saves the learned abstraction's weights.
          6. Rebuilds the Keras model via the composer.
          7. Fine-tunes the composed model for 300 epochs (only the output neuron's weights should change).
          8. Evaluates and prints the final accuracy.
          9. Verifies that the learned abstraction's weights are identical before and after fine-tuning.
          10. Stores the learned abstraction in the repository.
          11. Asserts that the final accuracy is above a chosen threshold.
        """
        # Create the environment with 500 examples per step.
        env = RLEnvironment(total_steps=1, num_instances_per_step=500, seed=0)
        env.reset()  # Generate the dataset now.
        features, true_labels = env.features, env.true_labels

        # Build the minimal network.
        composer, model = create_minimal_network(input_shape=(2,))
        
        # Train the learned abstraction on 500 examples.
        learned_abstraction = train_learned_abstraction_model(env, epochs=1000)
        
        # Freeze the learned abstraction so its weights remain unchanged.
        learned_abstraction.model.trainable = False
        weights_before = learned_abstraction.model.get_weights()

        # Add the learned abstraction into the network.
        composer.add_node(learned_abstraction)
        composer.connect("input", learned_abstraction.name)
        composer.connect(learned_abstraction.name, "output")
        composer.remove_connection("input", "output")

        # Rebuild the Keras model.
        model = composer.build()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Optionally visualize the graph here.
        # print_graph_nodes(composer)
        # visualize_graph(composer=composer)

        # Fine-tune the composed model.
        model.fit(features, true_labels, epochs=300, verbose=0)

        # Evaluate the fine-tuned model.
        loss, acc = model.evaluate(features, true_labels, verbose=0)
        print(f"Trained network accuracy after fine-tuning: {acc:.3f}")
        
        # Check that the learned abstraction's weights have not changed.
        weights_after = learned_abstraction.model.get_weights()
        for w_before, w_after in zip(weights_before, weights_after):
            assertion = "Learned abstraction weights should remain unchanged after fine-tuning."
            np.testing.assert_allclose(w_before, w_after, atol=1e-7, err_msg=assertion)
            print(assertion)
        
        # Store the learned abstraction in the repository.
        env.repository["learned_abstraction"] = learned_abstraction
        
        # Assert final accuracy is above threshold.
        assertion = "Trained network accuracy should be above 0.9."
        print(assertion)
        self.assertGreater(acc, 0.9, assertion)

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

#         # Define policies
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
