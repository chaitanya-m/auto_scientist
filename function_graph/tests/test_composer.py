# tests/test_composer.py
import unittest
import numpy as np
import keras
from graph.composer import GraphComposer
from graph.node import SingleNeuron, InputNode

class TestGraphComposer(unittest.TestCase):
    def test_train_graph_with_relu_and_sigmoid(self):
        composer = GraphComposer()

        # Define InputNode with explicit name
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
        
        # Pass input as a dictionary with key "input"
        x_dummy = np.random.rand(20, 10)
        y_dummy = np.random.rand(20, 1)
        history = model.fit({"input": x_dummy}, y_dummy, epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_two_relu_one_sigmoid(self):
        composer = GraphComposer()

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
        
        # Wire: input -> relu1/relu2 -> sigmoid
        composer.connect("input", "relu1", merge_mode='concat')
        composer.connect("input", "relu2", merge_mode='concat')
        composer.connect("relu1", "sigmoid", merge_mode='concat')
        composer.connect("relu2", "sigmoid", merge_mode='concat')
        
        model = composer.build()
        model.compile(optimizer="adam", loss="mse")
        
        # Input as dictionary with key "input"
        x_dummy = np.random.rand(20, 5)
        y_dummy = np.random.rand(20, 1)
        history = model.fit({"input": x_dummy}, y_dummy, epochs=1, verbose=0)
        self.assertIn("loss", history.history)

    def test_dense_graph_two_inputs_two_outputs(self):
        composer = GraphComposer()

        # Two InputNodes with unique names
        input1 = InputNode(name="input1", input_shape=(5,))
        input2 = InputNode(name="input2", input_shape=(5,))
        sigmoid1 = SingleNeuron(name="sigmoid1", activation="sigmoid")
        sigmoid2 = SingleNeuron(name="sigmoid2", activation="sigmoid")

        composer.add_node(input1)
        composer.add_node(input2)
        composer.add_node(sigmoid1)
        composer.add_node(sigmoid2)
        
        composer.set_input_node(["input1", "input2"])
        composer.set_output_node(["sigmoid1", "sigmoid2"])
        
        # Dense connections with explicit merge_mode
        for inp in ["input1", "input2"]:
            for out in ["sigmoid1", "sigmoid2"]:
                composer.connect(inp, out, merge_mode='concat')
        
        model = composer.build()
        model.compile(optimizer="adam", loss="mse")

        # Pass inputs as a dictionary with keys "input1" and "input2"
        x_dummy = np.random.rand(20, 5)
        y_dummy1 = np.random.rand(20, 1)
        y_dummy2 = np.random.rand(20, 1)
        history = model.fit(
            {"input1": x_dummy, "input2": x_dummy},  # Match input node names
            [y_dummy1, y_dummy2], 
            epochs=1, 
            verbose=0
        )
        self.assertIn("loss", history.history)

    def test_2x2_graph_vs_vanilla_keras(self):
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        composer = GraphComposer()

        # Input nodes with unique names
        input1 = InputNode(name="input1", input_shape=(2,))
        input2 = InputNode(name="input2", input_shape=(2,))
        out1 = SingleNeuron(name="out1", activation="linear")
        out2 = SingleNeuron(name="out2", activation="linear")

        composer.add_node(input1)
        composer.add_node(input2)
        composer.add_node(out1)
        composer.add_node(out2)
        
        composer.set_input_node(["input1", "input2"])
        composer.set_output_node(["out1", "out2"])
        
        # Dense connections
        for inp in ["input1", "input2"]:
            for out in ["out1", "out2"]:
                composer.connect(inp, out, merge_mode='concat')
        
        graph_model = composer.build()
        graph_model.compile(optimizer="adam", loss="mse")

        # Vanilla model
        inputs = keras.layers.Input(shape=(2,))
        outputs = keras.layers.Dense(2, activation="linear")(inputs)
        vanilla_model = keras.models.Model(inputs=inputs, outputs=outputs)
        vanilla_model.compile(optimizer="adam", loss="mse")

        # Train and evaluate with dictionary inputs
        N = 200
        x_data = np.random.rand(N, 2)
        y_data = 2 * x_data

        graph_model.fit(
            {"input1": x_data, "input2": x_data},  # Match input names
            [y_data[:, [0]], y_data[:, [1]]], 
            epochs=500, 
            verbose=0
        )
        vanilla_model.fit(x_data, y_data, epochs=500, verbose=0)

        # Evaluate with dictionary inputs
        graph_loss = graph_model.evaluate(
            {"input1": x_data, "input2": x_data}, 
            [y_data[:, [0]], y_data[:, [1]]], 
            verbose=0
        )
        vanilla_loss = vanilla_model.evaluate(x_data, y_data, verbose=0)

        if isinstance(graph_loss, list):
            graph_loss = sum(graph_loss) / len(graph_loss)

        self.assertLess(graph_loss, 0.2)
        self.assertLess(vanilla_loss, 0.2)
        self.assertAlmostEqual(graph_loss, vanilla_loss, delta=0.1)

if __name__ == "__main__":
    unittest.main()