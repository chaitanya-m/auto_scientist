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
    def test_single_neuron_blueprint(self):
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
