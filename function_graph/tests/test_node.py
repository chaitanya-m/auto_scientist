# tests/test_node.py
import unittest
import numpy as np
import keras
from function_graph.node import GraphNode, SingleNeuron

class TestGraphNodeAbstract(unittest.TestCase):
    def test_cannot_instantiate_graphnode(self):
        # GraphNode is abstract, so instantiating it should raise a TypeError.
        with self.assertRaises(TypeError):
            _ = GraphNode("dummy")

class TestSingleNeuron(unittest.TestCase):
    def test_build_node(self):
        # Create a SingleNeuron with a given activation.
        neuron = SingleNeuron(name="single_neuron", activation="relu")
        input_shape = (5,)  # For example, an input vector of length 5.
        neuron.build_node(input_shape)
        
        # Check that the model was built.
        self.assertIsNotNone(neuron.keras_model, "The keras_model should be built after calling build_node.")
        self.assertEqual(neuron.input_shape, input_shape, "The input_shape attribute should match the provided input shape.")
        self.assertEqual(neuron.output_shape, (1,), "Since SingleNeuron enforces units=1, the output shape should be (1,).")
    
    def test_apply_without_build(self):
        # Verify that calling apply() before build_node() raises a ValueError.
        neuron = SingleNeuron(name="single_neuron", activation="relu")
        dummy_input = np.random.rand(1, 5)  # 1 sample, 5 features.
        with self.assertRaises(ValueError):
            _ = neuron.apply(dummy_input)

    def test_apply_after_build(self):
        # After building, apply() should produce an output with the expected shape.
        neuron = SingleNeuron(name="single_neuron", activation="relu")
        input_shape = (5,)
        neuron.build_node(input_shape)
        dummy_input = np.random.rand(2, 5)  # 2 samples, 5 features each.
        output = neuron.apply(dummy_input)
        
        # The output shape should be (2, 1) (2 samples, 1 output per sample).
        self.assertEqual(output.shape, (2, 1), "The output shape should be (2, 1) after applying the model.")

if __name__ == '__main__':
    unittest.main()
