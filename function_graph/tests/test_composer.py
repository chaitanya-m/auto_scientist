# tests/test_composer.py
import unittest
import numpy as np
import keras
from graph.composer import GraphComposer
from graph.node import SingleNeuron, InputNode

class TestGraphComposer(unittest.TestCase):
    def test_train_graph_with_relu_and_sigmoid(self):
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

if __name__ == "__main__":
    unittest.main()
