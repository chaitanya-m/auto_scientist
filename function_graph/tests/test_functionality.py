# tests/test_functionality.py
import unittest
import numpy as np
import keras
from keras import layers
from graph.composer import GraphComposer
from graph.node import InputNode, SingleNeuron, SubGraphNode, GraphNode

# A helper dummy dense node for custom transformations.
class DummyDenseNode(GraphNode):
    def __init__(self, name, units, activation=None):
        super().__init__(name)
        self.units = units
        self.activation = activation

    def apply(self, input_tensor):
        return layers.Dense(self.units, activation=self.activation, name=self.name)(input_tensor)

class TestGraphComposerEdge(unittest.TestCase):
    def test_complex_graph_with_subgraph_shape_mismatch(self):
        """
        Build a complex graph where a saved subgraph (expecting 5 features)
        is reused in a main graph whose raw input has 8 features.
        A dummy dense node reduces 8 features to 5, and then the subgraph is applied.
        This tests node creation, connection, training, and prediction with shape adaptation.
        """
        # Create and save a simple subgraph that expects inputs with 5 features.
        sub_composer = GraphComposer()
        sub_input = InputNode(name="sub_input", input_shape=(5,))
        sub_neuron = SingleNeuron(name="sub_neuron", activation="linear")
        sub_composer.add_node(sub_input)
        sub_composer.add_node(sub_neuron)
        sub_composer.connect("sub_input", "sub_neuron", merge_mode='concat')
        sub_composer.set_input_node("sub_input")
        sub_composer.set_output_node("sub_neuron")
        sub_model = sub_composer.build()
        sub_model.compile(optimizer="adam", loss="mse")
        sub_composer.save_subgraph("complex_subgraph.keras")
        
        # Build the main graph with input shape of 8.
        main_composer = GraphComposer()
        main_input = InputNode(name="main_input", input_shape=(8,))
        # A dummy dense node that adapts features from 8 down to 5.
        reduce_node = DummyDenseNode(name="reduce", units=5, activation="linear")
        main_composer.add_node(main_input)
        main_composer.add_node(reduce_node)
        
        # Load the saved subgraph (expects 5 features) and add it.
        subgraph_node = SubGraphNode.load("complex_subgraph.keras", name="subgraph")
        main_composer.add_node(subgraph_node)
        
        # Wire the graph: main_input -> reduce -> subgraph.
        main_composer.connect("main_input", "reduce", merge_mode='concat')
        main_composer.connect("reduce", "subgraph", merge_mode='concat')
        main_composer.set_input_node("main_input")
        main_composer.set_output_node("subgraph")
        
        main_model = main_composer.build()
        main_model.compile(optimizer="adam", loss="mse")
        
        # Train on dummy data and verify predictions.
        x_data = np.random.rand(30, 8)
        y_data = np.random.rand(30, 1)
        history = main_model.fit(x_data, y_data, epochs=5, verbose=0)
        self.assertIn("loss", history.history)
        preds = main_model.predict(x_data)
        self.assertEqual(preds.shape, (30, 1))

    def test_simple_complex_graph_multiple_branches(self):
        """
        Create a graph with one input and two parallel branches that are merged into two outputs.
        This test simulates a scenario where one input feeds into two different nodes,
        and the outputs are then compared with a vanilla Keras model.
        """
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        composer = GraphComposer()

        # Single input node.
        input_node = InputNode(name="input", input_shape=(2,))
        out1 = SingleNeuron(name="out1", activation="linear")
        out2 = SingleNeuron(name="out2", activation="linear")

        composer.add_node(input_node)
        composer.add_node(out1)
        composer.add_node(out2)
        
        # One input node with two outputs.
        composer.set_input_node("input")
        composer.set_output_node(["out1", "out2"])
        
        # Wire the branches: input -> out1 and input -> out2.
        composer.connect("input", "out1", merge_mode='concat')
        composer.connect("input", "out2", merge_mode='concat')
        
        graph_model = composer.build()
        graph_model.compile(optimizer="adam", loss="mse")

        # Vanilla Keras model with a single input and two outputs.
        inputs = keras.layers.Input(shape=(2,))
        outputs = keras.layers.Dense(2, activation="linear")(inputs)
        vanilla_model = keras.models.Model(inputs=inputs, outputs=outputs)
        vanilla_model.compile(optimizer="adam", loss="mse")

        N = 200
        x_data = np.random.rand(N, 2)
        y_data = 2 * x_data

        graph_model.fit(x_data, [y_data[:, [0]], y_data[:, [1]]], epochs=500, verbose=0)
        vanilla_model.fit(x_data, y_data, epochs=500, verbose=0)

        graph_loss = graph_model.evaluate(x_data, [y_data[:, [0]], y_data[:, [1]]], verbose=0)
        vanilla_loss = vanilla_model.evaluate(x_data, y_data, verbose=0)

        if isinstance(graph_loss, list):
            graph_loss = sum(graph_loss) / len(graph_loss)
        self.assertLess(graph_loss, 0.2)
        self.assertLess(vanilla_loss, 0.2)
        self.assertAlmostEqual(graph_loss, vanilla_loss, delta=0.1)

class TestSubGraphEdge(unittest.TestCase):
    def test_complex_subgraph_reuse_edge_case(self):
        """
        Create a multi-layer subgraph with input shape 4 and save it.
        Then, load and reuse it in a main graph that provides an input with 7 features,
        triggering the shape adapter. Verify training and prediction work correctly.
        """
        # Build the complex subgraph.
        sub_composer = GraphComposer()
        sub_input = InputNode(name="sub_input", input_shape=(4,))
        hidden1 = DummyDenseNode("hidden1", units=6, activation="relu")
        hidden2 = DummyDenseNode("hidden2", units=4, activation="relu")
        out_node = DummyDenseNode("out", units=1, activation="linear")
        
        sub_composer.add_node(sub_input)
        sub_composer.add_node(hidden1)
        sub_composer.add_node(hidden2)
        sub_composer.add_node(out_node)
        
        sub_composer.connect("sub_input", "hidden1", merge_mode='concat')
        sub_composer.connect("hidden1", "hidden2", merge_mode='concat')
        sub_composer.connect("hidden2", "out", merge_mode='concat')
        
        sub_composer.set_input_node("sub_input")
        sub_composer.set_output_node("out")
        
        sub_model = sub_composer.build()
        sub_model.compile(optimizer="adam", loss="mse")
        sub_composer.save_subgraph("complex_subgraph_edge.keras")
        
        # Main graph with input shape 7 (mismatched with subgraph input shape of 4).
        main_composer = GraphComposer()
        main_input = InputNode(name="main_input", input_shape=(7,))
        main_composer.add_node(main_input)
        subgraph_node = SubGraphNode.load("complex_subgraph_edge.keras", name="subgraph")
        main_composer.add_node(subgraph_node)
        
        main_composer.connect("main_input", "subgraph", merge_mode='concat')
        main_composer.set_input_node("main_input")
        main_composer.set_output_node("subgraph")
        
        main_model = main_composer.build()
        main_model.compile(optimizer="adam", loss="mse")
        
        x_data = np.random.rand(50, 7)
        y_data = np.random.rand(50, 1)
        history = main_model.fit(x_data, y_data, epochs=5, verbose=0)
        self.assertIn("loss", history.history)
        preds = main_model.predict(x_data)
        self.assertEqual(preds.shape, (50, 1))

if __name__ == "__main__":
    unittest.main()
