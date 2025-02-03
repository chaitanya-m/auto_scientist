# tests/test_composer.py
import unittest
import numpy as np
import keras
from function_graph.composer import GraphComposer
from function_graph.node import SingleNeuron

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

if __name__ == "__main__":
    unittest.main()
