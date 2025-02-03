# tests/test_composer.py
import unittest
import numpy as np
import keras
from graph.composer import GraphComposer
from graph.node import SingleNeuron

class TestGraphComposer(unittest.TestCase):
    def test_train_graph_with_relu_and_sigmoid(self):
        # Create a GraphComposer instance.
        composer = GraphComposer()

        # Create a ReLU neuron and a Sigmoid neuron.
        # Note: Do not pass the "units" parameter since SingleNeuron always enforces units=1.
        relu_neuron = SingleNeuron(name="relu_neuron", activation="relu")
        sigmoid_neuron = SingleNeuron(name="sigmoid_neuron", activation="sigmoid")
        
        # Add the nodes to the composer.
        composer.add_node(relu_neuron)
        composer.add_node(sigmoid_neuron)
        
        # Designate the input and output nodes.
        composer.set_input_node("relu_neuron")
        composer.set_output_node("sigmoid_neuron")
        
        # Connect the nodes: relu_neuron -> sigmoid_neuron.
        composer.connect("relu_neuron", "sigmoid_neuron")
        
        # Build the complete Keras model.
        # Assume the global input has shape (10,) (i.e. 10 features, excluding the batch dimension).
        model = composer.build(input_shape=(10,))
        
        # Compile the model.
        model.compile(optimizer="adam", loss="mse")
        
        # Create dummy training data.
        x_dummy = np.random.rand(20, 10)  # 20 samples, 10 features each.
        y_dummy = np.random.rand(20, 1)   # 20 target values.
        
        # Train the model for one epoch.
        history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
        
        # Verify that the training history contains a loss value.
        self.assertIn("loss", history.history)
        
        # Verify that the model produces predictions of the expected shape.
        predictions = model.predict(x_dummy)
        self.assertEqual(predictions.shape, (20, 1))


    def test_two_relu_one_sigmoid(self):
        """
        Constructs a graph where two ReLU SingleNeuron nodes feed into a Sigmoid SingleNeuron node.
        The designated input node is one of the ReLU nodes.
        """
        composer = GraphComposer()

        # Create two ReLU neurons and one Sigmoid neuron.
        relu1 = SingleNeuron(name="relu1", activation="relu")
        relu2 = SingleNeuron(name="relu2", activation="relu")
        sigmoid = SingleNeuron(name="sigmoid", activation="sigmoid")

        # Add all nodes.
        composer.add_node(relu1)
        composer.add_node(relu2)
        composer.add_node(sigmoid)
        
        # Designate relu1 as the input node; the output node is the sigmoid.
        composer.set_input_node("relu1")
        composer.set_output_node("sigmoid")
        
        # Connect both ReLU nodes to the Sigmoid node.
        composer.connect("relu1", "sigmoid")
        composer.connect("relu2", "sigmoid")
        
        # Build the overall model. Global input is assumed to be 5-dimensional.
        model = composer.build(input_shape=(5,))
        model.compile(optimizer="adam", loss="mse")
        
        # Create dummy data: 20 samples, each with 5 features.
        x_dummy = np.random.rand(20, 5)
        # Target is one output per sample.
        y_dummy = np.random.rand(20, 1)
        
        # Train for one epoch.
        history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
        self.assertIn("loss", history.history)
        
        # Forward pass.
        predictions = model.predict(x_dummy)
        # Regardless of the two upstream nodes, the final sigmoid produces one output per sample.
        self.assertEqual(predictions.shape, (20, 1))

    def test_dense_graph_two_inputs_two_outputs(self):
        """
        Constructs a dense graph where two input ReLU SingleNeuron nodes feed into 
        two output Sigmoid SingleNeuron nodes. Dense means that every input neuron 
        is connected to every output neuron.
        """
        composer = GraphComposer()

        # Create two ReLU neurons (input neurons) and two Sigmoid neurons (output neurons)
        relu1 = SingleNeuron(name="relu1", activation="relu")
        relu2 = SingleNeuron(name="relu2", activation="relu")
        sigmoid1 = SingleNeuron(name="sigmoid1", activation="sigmoid")
        sigmoid2 = SingleNeuron(name="sigmoid2", activation="sigmoid")

        # Add all nodes to the composer.
        composer.add_node(relu1)
        composer.add_node(relu2)
        composer.add_node(sigmoid1)
        composer.add_node(sigmoid2)

        # Designate one of the ReLU nodes as the input node.
        # Note: Nodes without any parent connections will automatically receive the global input.
        composer.set_input_node("relu1")
        # Designate both sigmoid neurons as output nodes.
        composer.set_output_node(["sigmoid1", "sigmoid2"])

        # Dense connections: connect every ReLU to every Sigmoid.
        composer.connect("relu1", "sigmoid1")
        composer.connect("relu1", "sigmoid2")
        composer.connect("relu2", "sigmoid1")
        composer.connect("relu2", "sigmoid2")

        # Build the overall Keras model.
        # Assume a global input shape of (5,) i.e. each sample is a 5-element vector.
        model = composer.build(input_shape=(5,))
        model.compile(optimizer="adam", loss="mse")

        # Create dummy data: 20 samples, each with 5 features.
        x_dummy = np.random.rand(20, 5)
        # For multiple outputs, provide a list of target arrays (each with 20 samples and 1 output).
        y_dummy1 = np.random.rand(20, 1)
        y_dummy2 = np.random.rand(20, 1)

        # Train for one epoch.
        history = model.fit(x_dummy, [y_dummy1, y_dummy2], epochs=1, verbose=0)
        self.assertIn("loss", history.history)

        # Forward pass: predictions should be a list of two arrays.
        predictions = model.predict(x_dummy)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0].shape, (20, 1))
        self.assertEqual(predictions[1].shape, (20, 1))



if __name__ == "__main__":
    unittest.main()
