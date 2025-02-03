# node.py
from abc import ABC, abstractmethod
import keras

class GraphNode(ABC):
    """
    Represents a node in the computation graph.
    Each node encapsulates a small neural network that is built using
    the Keras Functional API when the graph is assembled.
    
    Attributes:
        name (str): The unique name of the node.
        input_shape (tuple): The expected input shape (excluding the batch dimension).
        output_shape (tuple): The output shape produced by the node (excluding the batch dimension).
        keras_model (keras.Model): The Keras model representing the node's computation.
    """
    def __init__(self, name: str):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.keras_model = None

    @abstractmethod
    def build_node(self, input_shape):
        """
        Given an input shape (excluding the batch dimension), build the underlying
        Keras model using the Functional API.
        
        Args:
            input_shape (tuple): The shape of the input (excluding the batch dimension).
        """
        pass

    def apply(self, input_tensor):
        """
        Applies the node's underlying Keras model to the input tensor.
        
        Args:
            input_tensor: The input tensor.
        
        Returns:
            The output tensor produced by the node.
        
        Raises:
            ValueError: If the node has not yet been built.
        """
        if self.keras_model is None:
            raise ValueError(f"Node '{self.name}' has not been built yet.")
        return self.keras_model(input_tensor)


class SingleNeuron(GraphNode):
    """
    A concrete node that encapsulates a single Dense layer (a single neuron).
    This class enforces that the neuron produces exactly one output.
    """
    def __init__(self, name: str, activation=None):
        """
        Initializes the SingleNeuron node with a single output unit.
        
        Args:
            name (str): The unique name for the node.
            activation: The activation function to use (e.g., 'relu', 'sigmoid'). Defaults to None.
        """
        super().__init__(name)
        self.activation = activation
        self.units = 1  # Enforce that this node always has a single output.

    def build_node(self, input_shape):
        """
        Builds the underlying Keras model for this node using the Functional API.
        
        Args:
            input_shape (tuple): The input shape (excluding the batch dimension).
        """
        self.input_shape = input_shape
        inp = keras.layers.Input(shape=input_shape, name=f"{self.name}_input")
        out = keras.layers.Dense(self.units, activation=self.activation, name=self.name)(inp)
        self.keras_model = keras.models.Model(inputs=inp, outputs=out, name=self.name)
        self.output_shape = self.keras_model.output_shape[1:]
