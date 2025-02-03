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

class InputNode(GraphNode):
    """
    A node that represents an input. It does not apply any transformations.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def apply(self, input_tensor):
        """
        Input nodes do not modify input tensors—they simply pass them forward.
        """
        return input_tensor  # Directly return input without applying a layer

class SubGraphNode(GraphNode):
    """
    Blueprint for a subgraph node. This allows a saved subgraph
    to be integrated as a hidden node in any new graph without shape mismatches.
    """
    def __init__(self, name: str, model):
        super().__init__(name)
        self.model = model

    def apply(self, input_tensor):
        """
        Given an input tensor, rebuild the subgraph so that its internal Input layer is
        replaced by the provided tensor. We assume that self.model was saved so that its
        first layer is an Input layer. Here we “skip” that layer and reapply the rest.
        """
        x = input_tensor
        for layer in self.model.layers[1:]:
            x = layer(x)
        return x

    @classmethod
    def load(cls, filepath, name, compile_model=False, optimizer="adam", loss="mse"):
        """
        Loads a saved subgraph (a Keras model) from a file and returns a SubGraphNode instance.

        Parameters:
            filepath (str): Path to the saved subgraph model.
            name (str): Name for the new SubGraphNode instance.
            compile_model (bool): If True, the model is compiled for training. Default is False (Inference only).
            optimizer (str): Optimizer to use if compiling the model.
            loss (str): Loss function to use if compiling the model.

        Returns:
            SubGraphNode: A new instance containing the loaded model.
        """
        loaded_model = keras.models.load_model(filepath.replace(".h5", ".keras"))
        
        if compile_model:
            loaded_model.compile(optimizer=optimizer, loss=loss)
        
        return cls(name, loaded_model)
