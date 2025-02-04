# graph/node.py
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
    def __init__(self, name: str, input_shape):
        super().__init__(name)
        self.input_shape = input_shape

    def apply(self, input_tensor):
        """
        Input nodes do not modify input tensors—they simply pass them forward.
        """
        return input_tensor  # Directly return input without applying a layer

class SubGraphNode(GraphNode):
    """
    Blueprint for a subgraph node. This allows a saved subgraph
    to be integrated as a hidden node in any new graph without shape mismatches.

    Problem:
    - If a full saved model is used in a new graph, it expects raw input data, 
      but in a new graph, it should receive activations from previous layers.
    - Using `self.model(input_tensor)` directly would assume the subgraph's Input layer 
      is still valid, leading to shape mismatches.

    Solution:
    - The `apply()` method **skips** the saved Input layer.
    - Instead, it **takes an incoming tensor** and applies only the internal layers.
    - This allows the subgraph to act as a hidden layer in any architecture.

    Attributes:
        name (str): Name of the subgraph node.
        model (keras.Model): The Keras model representing the subgraph.


    """
    def __init__(self, name: str, model):
        """
        Initializes a SubGraphNode with a pre-loaded Keras model.

        Args:
            name (str): Name of the subgraph node.
            model (keras.Model): The pre-loaded subgraph model.
        """
        super().__init__(name)
        self.model = model


    def apply(self, input_tensor):
        """
        Applies the subgraph to an incoming tensor, inserting a learned shape adapter
        if the incoming feature dimension does not match the subgraph's expected input.

        Args:
            input_tensor (tf.Tensor): The tensor representing input activations from a previous layer.

        Returns:
            tf.Tensor: The transformed tensor after passing through the subgraph.
        """
        x = input_tensor
        # Handle different input structures
        if isinstance(self.model.input, dict):
            # Get first input's shape (assuming single input subgraph)
            input_key = next(iter(self.model.input.keys()))
            expected_units = self.model.input[input_key].shape[1]
        elif isinstance(self.model.input, list):
            # Get first input's shape
            expected_units = self.model.input[0].shape[1]
        else:
            # Single input tensor
            expected_units = self.model.input.shape[1]

        # Add shape adapter if needed
        if x.shape[1] != expected_units:
            x = layers.Dense(expected_units, activation="linear", 
                           name=f"{self.name}_shape_adapter")(x)
        
        # Apply all layers except original Input layers
        for layer in self.model.layers[1:]:  # Skip input layer
            x = layer(x)
            
        return x


    @classmethod
    def load(cls, filepath, name, compile_model=False, optimizer="adam", loss="mse"):
        """
        Loads a saved subgraph model and returns it as a SubGraphNode.

        Problem:
        - The saved model should only be compiled if it will be used for training.
        - If the model is already compiled, redundant compilation can cause issues.

        Solution:
        - Allow an option (`compile_model`) to decide if the model should be compiled.
        - Check if the model already has an optimizer to avoid unnecessary recompilation.

        Args:
            filepath (str): Path to the saved subgraph model.
            name (str): Name for the new SubGraphNode instance.
            compile_model (bool): Whether to compile the model for training. Default is False.
            optimizer (str): The optimizer to use if compiling.
            loss (str): The loss function to use if compiling.

        Returns:
            SubGraphNode: A new instance containing the loaded model.
        """
        loaded_model = keras.models.load_model(filepath.replace(".h5", ".keras"))
        
        if compile_model: # Only compile if needed
            loaded_model.compile(optimizer=optimizer, loss=loss)
        
        return cls(name, loaded_model)
