# tests/test_subgraph.py
import unittest
import numpy as np
import keras
from graph.composer import GraphComposer
from graph.node import SingleNeuron, InputNode, SubGraphNode

class TestSubGraphNode(unittest.TestCase):

    def setUp(self):
        # Build model with explicit input naming
        composer = GraphComposer()
        input_node = InputNode(name="main_input", input_shape=(5,))
        relu = SingleNeuron(name="relu", activation="relu")
        
        composer.add_node(input_node)
        composer.add_node(relu)
        composer.connect("main_input", "relu", merge_mode='concat')
        
        composer.set_input_node("main_input")
        composer.set_output_node("relu")
        
        self.model = composer.build()
        self.model.compile(optimizer="adam", loss="mse")
        composer.save_subgraph("test_subgraph.keras")

    def test_subgraph_in_new_graph(self):
        subgraph_node = SubGraphNode.load("test_subgraph.keras", name="subgraph")
        
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input", input_shape=(5,))
        new_composer.add_node(new_input)
        new_composer.add_node(subgraph_node)
        
        # Explicit connection using original input name
        new_composer.connect("new_input", "subgraph", merge_mode='concat')
        
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("subgraph")
        
        new_model = new_composer.build()
        new_model.compile(optimizer="adam", loss="mse")
        
        # Test with proper input format
        x_test = np.random.rand(10, 5)
        predictions = new_model.predict(x_test)
        self.assertEqual(predictions.shape, (10, 1))


    def test_subgraph_loading(self):
        # Should load without errors
        subgraph_node = SubGraphNode.load("test_subgraph.keras", name="subgraph", compile_model=True)
        self.assertIsInstance(subgraph_node.model, keras.models.Model)

    def test_subgraph_reuse_fewer_features(self):
        subgraph_node = SubGraphNode.load("test_subgraph.keras", name="subgraph", compile_model=True)
        
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input", input_shape=(3,))
        new_composer.add_node(new_input)
        new_composer.add_node(subgraph_node)
        new_composer.connect("new_input", "subgraph", merge_mode='concat')
        
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("subgraph")
        
        # Should handle shape adaptation automatically
        new_model = new_composer.build()
        predictions = new_model.predict(np.random.rand(10, 3))
        self.assertEqual(predictions.shape, (10, 1))

    def test_subgraph_reuse_more_features(self):
        subgraph_node = SubGraphNode.load("test_subgraph.keras", name="subgraph", compile_model=True)
        
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input", input_shape=(7,))
        new_composer.add_node(new_input)
        new_composer.add_node(subgraph_node)
        new_composer.connect("new_input", "subgraph", merge_mode='concat')
        
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("subgraph")
        
        # Should handle shape adaptation automatically
        new_model = new_composer.build()
        predictions = new_model.predict(np.random.rand(10, 7))
        self.assertEqual(predictions.shape, (10, 1))

    def test_train_subgraph_in_new_graph(self):
        subgraph_node = SubGraphNode.load("test_subgraph.keras", name="subgraph", compile_model=True)
        
        new_composer = GraphComposer()
        new_input = InputNode(name="new_input", input_shape=(5,))
        new_composer.add_node(new_input)
        new_composer.add_node(subgraph_node)
        new_composer.connect("new_input", "subgraph", merge_mode='concat')
        
        new_composer.set_input_node("new_input")
        new_composer.set_output_node("subgraph")
        
        new_model = new_composer.build()
        new_model.compile(optimizer="adam", loss="mse")
        
        # Basic training test
        history = new_model.fit(
            np.random.rand(20, 5),
            np.random.rand(20, 1),
            epochs=1,
            verbose=0
        )
        self.assertIn("loss", history.history)

if __name__ == "__main__":
    unittest.main()