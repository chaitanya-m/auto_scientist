from abc import ABC, abstractmethod
import keras
from keras import layers

class GraphNode(ABC):
    """
    Blueprint for a node in the computation graph.
    Rather than wrapping an already-instantiated Keras model,
    this class specifies a function (via its apply() method) that,
    given an input tensor, returns an output tensor built using Keras layers.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, input_tensor):
        """
        Given an input tensor, build and return the output tensor.
        """
        pass

class SingleNeuron(GraphNode):
    """
    Blueprint for a single neuron: a Dense layer with one unit.
    The activation is specified as a parameter.
    (This enforces units=1.)
    """
    def __init__(self, name: str, activation=None):
        super().__init__(name)
        self.activation = activation

    def apply(self, input_tensor):
        # Each time this blueprint is used, it creates a new Dense layer.
        return layers.Dense(1, activation=self.activation, name=self.name)(input_tensor)
