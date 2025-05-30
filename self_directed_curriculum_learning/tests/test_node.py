# tests/test_node.py
import unittest
import numpy as np
import keras
from graph.node import GraphNode, SingleNeuron

class DummyNode(GraphNode):
    def apply(self, input_tensor):
        # Simply return the input tensor unchanged.
        return input_tensor

class TestSingleNeuronBlueprint(unittest.TestCase):
    """
    Unit tests for the SingleNeuron blueprint.
    """

    def test_single_neuron_blueprint(self):
        """
        This test case validates that a SingleNeuron instance configured with a ReLU
        activation produces an output of the expected shape.

        The test constructs a SingleNeuron and applies it to a global input tensor.
        A temporary Keras model is built to ensure that the output tensor has the
        expected shape, specifically (None, 1) as SingleNeuron outputs a single unit.
        """
        # Create a SingleNeuron blueprint with a ReLU activation.
        neuron = SingleNeuron(name="test_neuron", activation="relu")
        # Create a global input tensor.
        inp = keras.layers.Input(shape=(4,))
        # Apply the neuron blueprint.
        out = neuron.apply(inp)
        # Build a temporary model to check the output shape.
        model = keras.models.Model(inputs=inp, outputs=out)
        # Since SingleNeuron always outputs a single unit, we expect (None, 1).
        self.assertEqual(model.output_shape, (None, 1))
        
if __name__ == '__main__':
    unittest.main()
