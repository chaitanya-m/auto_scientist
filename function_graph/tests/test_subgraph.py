#tests/test_subgraph.py
import unittest
import numpy as np
import keras
from graph.composer import GraphComposer
from graph.node import SingleNeuron, InputNode, SubGraphNode

class TestSubGraphNode(unittest.TestCase):

    def setUp(self):
        """Creates a simple graph and saves a subgraph for testing."""
        self.composer = GraphComposer()

        # Define nodes
        self.input_node = InputNode(name="input")
        self.hidden1 = SingleNeuron(name="hidden1", activation="relu")
        self.hidden2 = SingleNeuron(name="hidden2", activation="relu")
        self.output_node = SingleNeuron(name="output", activation="linear")

        # Add nodes to the graph
        self.composer.add_node(self.input_node)
        self.composer.add_node(self.hidden1)
        self.composer.add_node(self.hidden2)
        self.composer.add_node(self.output_node)

        # Define connections
        self.composer.set_input_node("input")
        self.composer.set_output_node("output")
        self.composer.connect("input", "hidden1")
        self.composer.connect("hidden1", "hidden2")
        self.composer.connect("hidden2", "output")

        # Build and save the subgraph (intermediate hidden layers)
        self.composer.build(input_shape=(5,))
        self.composer.save_subgraph("test_subgraph.keras")

    def test_subgraph_loading(self):
        """Tests if a saved subgraph can be loaded successfully."""
        loaded_subgraph = SubGraphNode.load("test_subgraph.keras", name="loaded_subgraph")
        self.assertIsInstance(loaded_subgraph, SubGraphNode)
        self.assertIsNotNone(loaded_subgraph.model)
        print("Subgraph loaded successfully.")

    def test_subgraph_output_consistency(self):
        """Ensures that the loaded subgraph produces the same output as the original."""
        loaded_subgraph = SubGraphNode.load("test_subgraph.keras", name="loaded_subgraph")

        # Generate dummy input
        x_dummy = np.random.rand(10, 5)  # 10 samples, 5 features

        # Compute original model output
        original_model = self.composer.keras_model
        original_output = original_model.predict(x_dummy)

        # Compute subgraph output
        subgraph_output = loaded_subgraph.model.predict(x_dummy)

        # Ensure the outputs are similar
        np.testing.assert_almost_equal(original_output, subgraph_output, decimal=5)
        print("Subgraph produces the same output as the original.")

    def test_subgraph_in_new_graph(self):
        """Tests if a loaded subgraph can be used as a node in a new graph."""
        new_composer = GraphComposer()

        # Define new input and output nodes
        new_input = InputNode(name="new_input")
        new_output = SingleNeuron(name="new_output", activation="linear")

        # Load the saved subgraph as a node
        loaded_subgraph = SubGraphNode.load("test_subgraph.keras", name="reused_subgraph")

        # Add nodes to new graph
        new_composer.add_node(new_input)
        new_composer.add_node(loaded_subgraph)
        new_composer.add_node(new_output)

        # Connect nodes
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("new_output")
        new_composer.connect("new_input", "reused_subgraph")
        new_composer.connect("reused_subgraph", "new_output")

        # Build the new graph
        new_model = new_composer.build(input_shape=(5,))
        self.assertIsInstance(new_model, keras.models.Model)
        print("Subgraph successfully integrated into a new graph.")

    def test_train_subgraph_in_new_graph(self):
        """Tests if a new graph containing a subgraph node can be trained."""
        new_composer = GraphComposer()

        # Define new input and output nodes
        new_input = InputNode(name="new_input")
        new_output = SingleNeuron(name="new_output", activation="linear")

        # Load the saved subgraph as a node
        loaded_subgraph = SubGraphNode.load("test_subgraph.keras", name="reused_subgraph")

        # Add nodes to new graph
        new_composer.add_node(new_input)
        new_composer.add_node(loaded_subgraph)
        new_composer.add_node(new_output)

        # Connect nodes
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("new_output")
        new_composer.connect("new_input", "reused_subgraph")
        new_composer.connect("reused_subgraph", "new_output")

        # Build and compile the model
        new_model = new_composer.build(input_shape=(5,))
        new_model.compile(optimizer="adam", loss="mse")

        # Create dummy data
        x_train = np.random.rand(20, 5)  # 20 samples, 5 features
        y_train = np.random.rand(20, 1)  # 20 target values

        # Train the model
        history = new_model.fit(x_train, y_train, epochs=5, verbose=0)

        # Ensure training loss is decreasing
        final_loss = history.history["loss"][-1]
        self.assertLess(final_loss, history.history["loss"][0])
        print("Subgraph successfully trained in a new graph.")


    def test_subgraph_reuse_fewer_features(self):
        """
        Test reusing a stored subgraph in a new graph where the new input has fewer features
        than the subgraph's original expected input (e.g., 3 instead of 5).
        The shape adapter should adapt the 3-feature input to the expected 5 features.
        """
        # Create a simple model and train it on a trivial task (input shape 5)
        composer = GraphComposer()
        input_node = InputNode(name="input")
        neuron1 = SingleNeuron(name="neuron1", activation="relu")
        neuron2 = SingleNeuron(name="neuron2", activation="relu")
        output_node = SingleNeuron(name="output", activation="linear")
        composer.add_node(input_node)
        composer.add_node(neuron1)
        composer.add_node(neuron2)
        composer.add_node(output_node)
        composer.set_input_node("input")
        composer.set_output_node("output")
        composer.connect("input", "neuron1")
        composer.connect("neuron1", "neuron2")
        composer.connect("neuron2", "output")
        model = composer.build(input_shape=(5,))
        x_train = np.ones((10, 5))
        y_train = np.ones((10, 1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=2, verbose=0)
        composer.save_subgraph("temp_subgraph.keras")

        # Load the subgraph
        loaded_subgraph = SubGraphNode.load("temp_subgraph.keras", name="reused_subgraph")
        # Build a new graph with input shape (3,) -- fewer features
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input")
        new_output = SingleNeuron(name="new_output", activation="linear")
        new_composer.add_node(new_input)
        new_composer.add_node(loaded_subgraph)
        new_composer.add_node(new_output)
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("new_output")
        new_composer.connect("new_input", "reused_subgraph")
        new_composer.connect("reused_subgraph", "new_output")
        new_model = new_composer.build(input_shape=(3,))
        # Verify that the output shape is still (None, 1)
        self.assertEqual(new_model.output_shape, (None, 1))
        print("Subgraph reused with fewer features successfully.")

    def test_subgraph_reuse_more_features(self):
        """
        Test reusing a stored subgraph in a new graph where the new input has more features
        than the subgraph's original expected input (e.g., 7 instead of 5).
        The shape adapter should adapt the 7-feature input to the expected 5 features.
        """
        # Create a simple model and train it on a trivial task (input shape 5)
        composer = GraphComposer()
        input_node = InputNode(name="input")
        neuron1 = SingleNeuron(name="neuron1", activation="relu")
        neuron2 = SingleNeuron(name="neuron2", activation="relu")
        output_node = SingleNeuron(name="output", activation="linear")
        composer.add_node(input_node)
        composer.add_node(neuron1)
        composer.add_node(neuron2)
        composer.add_node(output_node)
        composer.set_input_node("input")
        composer.set_output_node("output")
        composer.connect("input", "neuron1")
        composer.connect("neuron1", "neuron2")
        composer.connect("neuron2", "output")
        model = composer.build(input_shape=(5,))
        x_train = np.ones((10, 5))
        y_train = np.ones((10, 1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=2, verbose=0)
        composer.save_subgraph("temp_subgraph.keras")

        # Load the subgraph
        loaded_subgraph = SubGraphNode.load("temp_subgraph.keras", name="reused_subgraph")
        # Build a new graph with input shape (7,) -- more features
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input")
        new_output = SingleNeuron(name="new_output", activation="linear")
        new_composer.add_node(new_input)
        new_composer.add_node(loaded_subgraph)
        new_composer.add_node(new_output)
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("new_output")
        new_composer.connect("new_input", "reused_subgraph")
        new_composer.connect("reused_subgraph", "new_output")
        new_model = new_composer.build(input_shape=(7,))
        # Verify that the output shape is (None, 1)
        self.assertEqual(new_model.output_shape, (None, 1))
        print("Subgraph reused with more features successfully.")


if __name__ == "__main__":
    unittest.main()
