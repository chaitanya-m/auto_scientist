# tests/test_composer.py

"""
This module verifies that the GraphComposer can correctly build computational graphs
using various combinations of nodes (e.g., InputNode and SingleNeuron) with different
activation functions and wiring strategies. It includes tests for both simple and complex
graph structures, ensuring that the composed Keras models perform as expected.

Tests various cases in composing graphs, including subgraph reuse and integration, 
shape mismatches between nodes, automatic shape adaptation, and multi-branch architectures.

We use dummy nodes and saved subgraphs to simulate real-world scenarios where
graphs are composed, connected, trained, and used for predictions.
"""

import unittest
import numpy as np
import keras
from keras import layers, models
from env.graph.composer import GraphComposer, GraphHasher
from env.graph.node import InputNode, SingleNeuron, SubGraphNode, GraphNode


class TestGraphComposerClone(unittest.TestCase):
    def setUp(self):
        # Build a simple composer with one input, one SingleNeuron, and one SubGraphNode
        self.composer = GraphComposer()
        # Input node
        inp = InputNode(name='input', input_shape=(4,))
        self.composer.add_node(inp)
        self.composer.set_input_node('input')

        # SingleNeuron node
        sn = SingleNeuron(name='neuron', activation='linear')
        self.composer.add_node(sn)
        self.composer.connect('input', 'neuron')

                # Create a bypass_model for SubGraphNode using the Functional API to ensure .input is defined
        inp_layer = layers.Input(shape=(1,), name='bypass_in')
        dense_out = layers.Dense(2, activation='linear', name='bypass_dense')(inp_layer)
        bypass = models.Model(inputs=inp_layer, outputs=dense_out)
        # Initialize unique weights for bypass
        weights = [np.full_like(w, fill_value=idx+1) for idx, w in enumerate(bypass.get_weights())]
        bypass.set_weights(weights)

        sgn = SubGraphNode(name='subgraph', model=bypass)
        self.composer.add_node(sgn)
        self.composer.connect('neuron', 'subgraph')
        self.composer.set_output_node('subgraph')

        # Build and compile original model
        self.original_model = self.composer.build()
        self.original_model.compile(optimizer='sgd', loss='mse')
        # Set distinct weights in original_model
        orig_weights = [np.arange(w.size).reshape(w.shape) for w in self.original_model.get_weights()]
        self.original_model.set_weights(orig_weights)

    def test_clone_preserves_weights_and_structure(self):
        # Clone the composer
        cloned_composer = self.composer.clone()
        cloned_model = cloned_composer.keras_model

        # Models should not be the same object
        self.assertIsNot(cloned_model, self.original_model)

                # Check structure: same number of layers
        orig_layers = [layer.name for layer in self.original_model.layers]
        clone_layers = [layer.name for layer in cloned_model.layers]
        self.assertEqual(len(orig_layers), len(clone_layers))
        # Allow UUID suffix differences in layer names (prefix must match)
        for orig_name, clone_name in zip(orig_layers, clone_layers):
            if orig_name != clone_name:
                # Compare prefixes up to the last underscore
                orig_prefix = '_'.join(orig_name.split('_')[:-1])
                clone_prefix = '_'.join(clone_name.split('_')[:-1])
                self.assertEqual(orig_prefix, clone_prefix)

        # Check weights: values equal but not same object: values equal but not same object
        for orig_w, clone_w in zip(self.original_model.get_weights(), cloned_model.get_weights()):
            np.testing.assert_array_equal(orig_w, clone_w)
            self.assertIsNot(orig_w, clone_w)

        # Check composer metadata
        self.assertEqual(self.composer.input_node_name, cloned_composer.input_node_name)
        self.assertListEqual(self.composer.output_node_names, cloned_composer.output_node_names)
        self.assertDictEqual(self.composer.connections, cloned_composer.connections)

    def test_clone_nodes_are_deep_copies(self):
        cloned = self.composer.clone()
        # Node objects should not be the same
        for name in self.composer.nodes:
            self.assertIsNot(self.composer.nodes[name], cloned.nodes[name])


class TestGraphComposer(unittest.TestCase):
    """
    Unit tests for the GraphComposer class.

    These tests validate that the GraphComposer correctly constructs and compiles models
    with various graph topologies and node configurations.
    """

    def test_train_graph_with_relu_and_sigmoid(self):
        """
        Test training of a graph with a ReLU followed by a Sigmoid neuron.

        This test builds a graph where an input node feeds into a ReLU-activated neuron,
        which then feeds into a Sigmoid-activated neuron. The graph is compiled and trained
        using random data. The test asserts that the training history contains a loss value.
        """

        composer = GraphComposer()

        # Define a single InputNode with explicit name
        input_node = InputNode(name="input", input_shape=(10,))
        relu = SingleNeuron(name="relu", activation="relu")
        sigmoid = SingleNeuron(name="sigmoid", activation="sigmoid")

        composer.add_node(input_node)
        composer.add_node(relu)
        composer.add_node(sigmoid)
        
        composer.set_input_node("input")
        composer.set_output_node("sigmoid")
        
        # Wire: input -> relu -> sigmoid
        composer.connect("input", "relu", merge_mode='concat')
        composer.connect("relu", "sigmoid", merge_mode='concat')
        
        model = composer.build()
        model.compile(optimizer="adam", loss="mse")
        
        # Pass input as a single tensor (no dictionary since only one input is supported)
        x_dummy = np.random.rand(20, 10)
        y_dummy = np.random.rand(20, 1)
        history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_two_relu_one_sigmoid(self):
        """
        Test a graph with two parallel ReLU neurons merging into a Sigmoid neuron.

        In this test, the graph consists of one input node that feeds into two parallel
        ReLU-activated neurons. Their outputs are merged and connected to a Sigmoid-activated
        neuron. The model is compiled and trained, and the test checks that a loss value
        is present in the training history.
        """

        composer = GraphComposer()

        # Define a single input node
        input_node = InputNode(name="input", input_shape=(5,))
        relu1 = SingleNeuron(name="relu1", activation="relu")
        relu2 = SingleNeuron(name="relu2", activation="relu")
        sigmoid = SingleNeuron(name="sigmoid", activation="sigmoid")

        composer.add_node(input_node)
        composer.add_node(relu1)
        composer.add_node(relu2)
        composer.add_node(sigmoid)
        
        composer.set_input_node("input")
        composer.set_output_node("sigmoid")
        
        # Wire: input -> relu1 and input -> relu2; then merge relu1 and relu2 -> sigmoid.
        composer.connect("input", "relu1", merge_mode='concat')
        composer.connect("input", "relu2", merge_mode='concat')
        composer.connect("relu1", "sigmoid", merge_mode='concat')
        composer.connect("relu2", "sigmoid", merge_mode='concat')
        
        model = composer.build()
        model.compile(optimizer="adam", loss="mse")
        
        # Pass input as a single tensor.
        x_dummy = np.random.rand(20, 5)
        y_dummy = np.random.rand(20, 1)
        history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_dense_graph_two_outputs(self):
        """
        Test a dense graph configuration with one input node and two output nodes.

        This test constructs a graph where a single input node feeds directly into two
        distinct output nodes (both using Sigmoid activation). The model is compiled and
        trained with two corresponding output tensors. The test asserts that the training
        history contains a loss value.
        """
        composer = GraphComposer()

        # One InputNode; two output nodes derived directly from the same input.
        input_node = InputNode(name="input", input_shape=(5,))
        sigmoid1 = SingleNeuron(name="sigmoid1", activation="sigmoid")
        sigmoid2 = SingleNeuron(name="sigmoid2", activation="sigmoid")

        composer.add_node(input_node)
        composer.add_node(sigmoid1)
        composer.add_node(sigmoid2)
        
        # Set single input and two outputs.
        composer.set_input_node("input")
        composer.set_output_node(["sigmoid1", "sigmoid2"])
        
        # Wire: input -> sigmoid1 and input -> sigmoid2.
        composer.connect("input", "sigmoid1", merge_mode='concat')
        composer.connect("input", "sigmoid2", merge_mode='concat')
        
        model = composer.build()
        model.compile(optimizer="adam", loss="mse")

        # Pass input as a single tensor.
        x_dummy = np.random.rand(20, 5)
        y_dummy1 = np.random.rand(20, 1)
        y_dummy2 = np.random.rand(20, 1)
        history = model.fit(x_dummy, [y_dummy1, y_dummy2], epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_2x2_graph_vs_vanilla_keras(self):
        """
        Compare the performance of a 2x2 graph model against a vanilla Keras model.

        This test sets up a graph with one input node and two linear output nodes. It also
        creates a vanilla Keras model with a single Dense layer producing two outputs. Both
        models are trained on the same randomly generated dataset. The test asserts that
        the loss for both models is low (below 0.2) and that the losses are nearly equal.
        """
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        composer = GraphComposer()

        # One input node with shape (2,)
        input_node = InputNode(name="input", input_shape=(2,))
        out1 = SingleNeuron(name="out1", activation="linear")
        out2 = SingleNeuron(name="out2", activation="linear")

        composer.add_node(input_node)
        composer.add_node(out1)
        composer.add_node(out2)
        
        # Set single input and two outputs.
        composer.set_input_node("input")
        composer.set_output_node(["out1", "out2"])
        
        # Wire: input -> out1 and input -> out2.
        composer.connect("input", "out1", merge_mode='concat')
        composer.connect("input", "out2", merge_mode='concat')
        
        graph_model = composer.build()
        graph_model.compile(optimizer="adam", loss="mse")

        # Vanilla model: single input, two outputs.
        inputs = keras.layers.Input(shape=(2,))
        outputs = keras.layers.Dense(2, activation="linear")(inputs)
        vanilla_model = keras.models.Model(inputs=inputs, outputs=outputs)
        vanilla_model.compile(optimizer="adam", loss="mse")

        # Train and evaluate using a single input tensor.
        N = 200
        x_data = np.random.rand(N, 2)
        y_data = 2 * x_data

        graph_model.fit(x_data, [y_data[:, [0]], y_data[:, [1]]], epochs=500, verbose=0)
        vanilla_model.fit(x_data, y_data, epochs=500, verbose=0)

        # Evaluate
        graph_loss = graph_model.evaluate(x_data, [y_data[:, [0]], y_data[:, [1]]], verbose=0)
        vanilla_loss = vanilla_model.evaluate(x_data, y_data, verbose=0)

        if isinstance(graph_loss, list):
            graph_loss = sum(graph_loss) / len(graph_loss)

        self.assertLess(graph_loss, 0.2)
        self.assertLess(vanilla_loss, 0.2)
        self.assertAlmostEqual(graph_loss, vanilla_loss, delta=0.1)

    def test_complex_graph_with_subgraph(self):
        """
        Test a complex graph where we manually handle a shape mismatch between
        the main graph and a saved subgraph. In this test, the subgraph is designed
        to expect inputs with 5 features, while the main graph's raw input has 8 features.
        
        Instead of relying on an automatic adapter, we explicitly use a DummyDenseNode to
        reduce the 8 features to 5 before passing them to the subgraph. 
        
        This test verifies that manual shape adaptation works as intended and that the composed model can be
        built, trained, and used for prediction. Note: Automatic adaptation is tested separately.
        """
        # Create and save a simple subgraph that expects inputs with 5 features.
        sub_composer = GraphComposer()
        sub_input = InputNode(name="sub_input", input_shape=(5,))  # Subgraph input with 5 features.
        sub_neuron = SingleNeuron(name="sub_neuron", activation="linear")  # Single neuron with linear activation.
        sub_composer.add_node(sub_input)
        sub_composer.add_node(sub_neuron)
        sub_composer.connect("sub_input", "sub_neuron", merge_mode='concat')  # Connect sub_input to sub_neuron.
        sub_composer.set_input_node("sub_input")  # Set the subgraph input.
        sub_composer.set_output_node("sub_neuron")  # Set the subgraph output.
        sub_model = sub_composer.build()  # Build the subgraph model.
        sub_model.compile(optimizer="adam", loss="mse")  # Compile the subgraph model.
        sub_composer.save_subgraph("complex_subgraph.keras")  # Save the subgraph.
        
        # Build the main graph with input shape of 8.
        main_composer = GraphComposer()
        main_input = InputNode(name="main_input", input_shape=(8,))  # Main graph input with 8 features.
        # A dummy dense node that reduces features from 8 to 5.
        reduce_node = DummyDenseNode(name="reduce", units=5, activation="linear")
        main_composer.add_node(main_input)
        main_composer.add_node(reduce_node)
        
        # Load the saved subgraph (which expects 5 features) and add it.
        subgraph_node = SubGraphNode.load("complex_subgraph.keras", name="subgraph")
        main_composer.add_node(subgraph_node)
        
        # Wire the graph: main_input -> reduce -> subgraph.
        main_composer.connect("main_input", "reduce", merge_mode='concat')
        main_composer.connect("reduce", "subgraph", merge_mode='concat')
        main_composer.set_input_node("main_input")
        main_composer.set_output_node("subgraph")
        
        main_model = main_composer.build()  # Build the main model.
        main_model.compile(optimizer="adam", loss="mse")  # Compile the main model.
        
        # Train on dummy data and verify predictions.
        x_data = np.random.rand(30, 8)  # Generate 30 samples with 8 features.
        y_data = np.random.rand(30, 1)  # Generate corresponding targets.
        history = main_model.fit(x_data, y_data, epochs=5, verbose=0)
        self.assertIn("loss", history.history)  # Check that training recorded a loss.
        preds = main_model.predict(x_data)  # Get predictions.
        self.assertEqual(preds.shape, (30, 1))  # Verify output shape.

    def test_simple_complex_graph_multiple_branches(self):
        """
        Test a graph with one input splitting into two parallel branches and merging into two outputs.

        In this test, we construct a graph where a single input is fed into two independent branches,
        each processed by its own neuron, resulting in two distinct outputs. 
        
        The rationale behind this test is to validate that our GraphComposer can correctly manage multiple parallel branches and that
        the resulting model behaves as expected. 
        
        We compare the composed graph against a vanilla Keras model
        with an equivalent structure to ensure that both approaches achieve similar performance (i.e., low loss)
        on the same dataset. This comparison helps confirm that our custom graph-building mechanism is
        reliable and produces models with performance on par with standard Keras implementations.
        """

        np.random.seed(42)
        keras.utils.set_random_seed(42)

        composer = GraphComposer()

        # Define a single input node with 2 features.
        input_node = InputNode(name="input", input_shape=(2,))
        out1 = SingleNeuron(name="out1", activation="linear")  # First branch output.
        out2 = SingleNeuron(name="out2", activation="linear")  # Second branch output.

        composer.add_node(input_node)
        composer.add_node(out1)
        composer.add_node(out2)
        
        # Set the single input and two outputs.
        composer.set_input_node("input")
        composer.set_output_node(["out1", "out2"])
        
        # Wire the branches: input -> out1 and input -> out2.
        composer.connect("input", "out1", merge_mode='concat')
        composer.connect("input", "out2", merge_mode='concat')
        
        graph_model = composer.build()  # Build the composed graph.
        graph_model.compile(optimizer="adam", loss="mse")  # Compile the model.

        # Create a vanilla Keras model for comparison.
        inputs = keras.layers.Input(shape=(2,))
        outputs = keras.layers.Dense(2, activation="linear")(inputs)
        vanilla_model = keras.models.Model(inputs=inputs, outputs=outputs)
        vanilla_model.compile(optimizer="adam", loss="mse")

        # Generate dummy data.
        N = 200
        x_data = np.random.rand(N, 2)
        y_data = 2 * x_data  # Target is a scaled version of the input.

        # Train both models.
        graph_model.fit(x_data, [y_data[:, [0]], y_data[:, [1]]], epochs=500, verbose=0)
        vanilla_model.fit(x_data, y_data, epochs=500, verbose=0)

        # Evaluate both models.
        graph_loss = graph_model.evaluate(x_data, [y_data[:, [0]], y_data[:, [1]]], verbose=0)
        vanilla_loss = vanilla_model.evaluate(x_data, y_data, verbose=0)

        # If graph_loss is a list (multiple outputs), average it.
        if isinstance(graph_loss, list):
            graph_loss = sum(graph_loss) / len(graph_loss)
        self.assertLess(graph_loss, 0.2)  # Ensure loss is low.
        self.assertLess(vanilla_loss, 0.2)
        self.assertAlmostEqual(graph_loss, vanilla_loss, delta=0.1)  # Compare losses.


# A helper dummy dense node for custom transformations.
class DummyDenseNode(GraphNode):
    """
    A simple dummy node that wraps a Keras Dense layer.

    This node is used to simulate custom transformations in the graph. It creates
    a Dense layer with a specified number of units and an optional activation function.
    """
    def __init__(self, name, units, activation=None):
        """
        Initialize the DummyDenseNode.
        """
        super().__init__(name)  # Initialize base GraphNode with the given name.
        self.units = units      # Number of units for the Dense layer.
        self.activation = activation  # Activation function.
    
    def apply(self, input_tensor):
        # Return the output of a Dense layer applied to the input tensor.
        return layers.Dense(self.units, activation=self.activation, name=self.name)(input_tensor)
class TestSubGraph(unittest.TestCase):
    """
    Tests focusing on automatically reusing multi-layer subgraphs in different main graphs,
    including handling input shape mismatches via shape adapters.
    """

    def test_complex_subgraph_reuse_with_shape_mismatch(self):
        """

        Test reusing a multi-layer subgraph in a main graph with a mismatched input shape.

        First, a subgraph is built with an input of 4 features and multiple layers.
        The subgraph is saved to disk. 
        
        Then, a main graph is created with an input of 7 features. 
        
        When the saved subgraph is loaded, the system should automatically
        adapt the shape from 7 features to the expected 4 features. The test confirms that
        the main graph trains and predicts correctly.
        """
        # Build the complex subgraph.
        sub_composer = GraphComposer()
        sub_input = InputNode(name="sub_input", input_shape=(4,))  # Subgraph input: 4 features.
        hidden1 = DummyDenseNode("hidden1", units=6, activation="relu")
        hidden2 = DummyDenseNode("hidden2", units=4, activation="relu")
        out_node = DummyDenseNode("out", units=1, activation="linear")
        
        sub_composer.add_node(sub_input)
        sub_composer.add_node(hidden1)
        sub_composer.add_node(hidden2)
        sub_composer.add_node(out_node)
        
        # Wire the subgraph: sub_input -> hidden1 -> hidden2 -> out.
        sub_composer.connect("sub_input", "hidden1", merge_mode='concat')
        sub_composer.connect("hidden1", "hidden2", merge_mode='concat')
        sub_composer.connect("hidden2", "out", merge_mode='concat')
        
        sub_composer.set_input_node("sub_input")
        sub_composer.set_output_node("out")
        
        sub_model = sub_composer.build()  # Build subgraph model.
        sub_model.compile(optimizer="adam", loss="mse")
        sub_composer.save_subgraph("complex_subgraph_edge.keras")  # Save the subgraph.
        
        # Build the main graph with a mismatched input shape (7 instead of 4).
        main_composer = GraphComposer()
        main_input = InputNode(name="main_input", input_shape=(7,))  # Main input: 7 features.
        main_composer.add_node(main_input)
        # Load the saved subgraph, which expects 4 features.
        subgraph_node = SubGraphNode.load("complex_subgraph_edge.keras", name="subgraph")
        main_composer.add_node(subgraph_node)
        
        # Wire the main graph: main_input -> subgraph.
        main_composer.connect("main_input", "subgraph", merge_mode='concat')
        main_composer.set_input_node("main_input")
        main_composer.set_output_node("subgraph")
        
        main_model = main_composer.build()  # Build main model.
        main_model.compile(optimizer="adam", loss="mse")
        
        # Train the main model with dummy data.
        x_data = np.random.rand(50, 7)
        y_data = np.random.rand(50, 1)
        history = main_model.fit(x_data, y_data, epochs=5, verbose=0)
        self.assertIn("loss", history.history)
        preds = main_model.predict(x_data)
        self.assertEqual(preds.shape, (50, 1))  # Verify prediction shape.


class TestGraphHash(unittest.TestCase):
    def setUp(self):
        # Create an instance of GraphComposer and build a simple graph.
        self.composer = GraphComposer()
        # Add an InputNode and a SingleNeuron node.
        self.composer.add_node(InputNode(name="input", input_shape=(3,)))
        self.composer.add_node(SingleNeuron(name="neuron1", activation="relu"))
        # Set the input and output nodes.
        self.composer.set_input_node("input")
        self.composer.set_output_node("neuron1")
        # Connect the input node to the neuron.
        self.composer.connect("input", "neuron1", merge_mode="concat")

    def test_hash_stability(self):
        # Ensure that the same hash is produced for an unchanged graph.
        hash1 = GraphHasher.hash(self.composer)
        hash2 = GraphHasher.hash(self.composer)
        self.assertEqual(hash1, hash2, "Hash should be stable for unchanged graphs.")

    def test_hash_change_with_node_addition(self):
        # Save the original hash.
        original_hash = GraphHasher.hash(self.composer)
        # Add a new node and connect it.
        new_node_name = "neuron2"
        self.composer.add_node(SingleNeuron(name=new_node_name, activation="sigmoid"))
        self.composer.connect("input", new_node_name, merge_mode="concat")
        # Get the new hash.
        new_hash = GraphHasher.hash(self.composer)
        self.assertNotEqual(original_hash, new_hash, "Hash should change when a new node is added.")

    def test_hash_change_with_connection_modification(self):
        # Save the original hash.
        original_hash = GraphHasher.hash(self.composer)
        # Modify the connection: remove the existing connection and add a new one with a different merge_mode.
        self.composer.remove_connection("input", "neuron1", merge_mode="concat")
        self.composer.connect("input", "neuron1", merge_mode="add")
        # Get the new hash.
        new_hash = GraphHasher.hash(self.composer)
        self.assertNotEqual(original_hash, new_hash, "Hash should change when connection merge mode is modified.")


if __name__ == "__main__":
    unittest.main()

