# graph/node.py
from abc import ABC, abstractmethod
import keras
from keras import layers
import uuid

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
        return layers.Dense(1, activation=self.activation, name=self.name)(input_tensor)

class InputNode(GraphNode):
    """
    A node that represents an input. It does not apply any transformations.
    """
    def __init__(self, name: str, input_shape):
        super().__init__(name)
        self.input_shape = input_shape

    def apply(self, input_tensor):
        return input_tensor

class SubGraphNode(GraphNode):
    """
    Blueprint for a subgraph node. This allows a saved subgraph to be integrated as a hidden node
    in any new graph without shape mismatches. This version assumes a single input only,
    eliminating multimodal/multi-input handling.
    
    The saved subgraph model is expected to have a single Input layer.
    """
    def __init__(self, name: str, model):
        """
        Initializes a SubGraphNode with a pre-loaded Keras model and builds a bypass model.

        We need a bypass model because the saved subgraph model comes with its own predefined Input layer, 
        which is set up for receiving raw data in isolation. When you integrate that saved model as a subgraph 
        (or hidden node) into a larger, composed graph, you don't want to use its original Input layer — you want 
        to feed it the activations coming from the previous layers of the new model. By creating a bypass model, 
        we replace the original Input layer with a new one that matches the expected shape (excluding the batch 
        dimension) but is part of the new overall graph. This allows the subgraph to operate seamlessly as a 
        hidden component of the larger model while preserving its internal architecture and learned weights.

        Args:
            name (str): Name of the subgraph node.
            model (keras.Model): The pre-loaded subgraph model (with a single input).
        """
        super().__init__(name)  # Initialize the parent GraphNode with the given name
        self.model = model  # Store the original subgraph model

        # Create a unique input layer name to avoid conflicts.
        unique_input_name = f"{name}_bypass_input_{uuid.uuid4().hex[:6]}"

        new_input = keras.layers.Input(shape=model.input.shape[1:], name=unique_input_name)  # Create a new Input layer matching the original model's input shape (excluding batch size)
        x = self.model(new_input)  # Pass the new input through the original model to compute the output tensor
        self.bypass_model = keras.models.Model(new_input, x)  # Build the bypass model using the new input and its computed output


    def apply(self, input_tensor):
        """
        Applies the subgraph to an incoming tensor, automatically adapting its feature dimension
        to match the subgraph's expected input. A trainable Dense layer with linear activation is
        inserted when the incoming tensor’s feature count differs from the expected value. This
        mechanism adds minimal computational overhead while enhancing reusability, though larger
        mismatches may require extra training to achieve optimal performance.

        Args:
            input_tensor (tf.Tensor): A single tensor representing activations from a previous layer.
        
        Returns:
            tf.Tensor: The transformed tensor after shape adaptation (if needed) and processing
                       through the subgraph.
        """
        expected_units = self.model.input.shape[1]     # Get expected feature dimension from the original model's Input layer.
        if input_tensor.shape[1] != expected_units:    # Check if shape adaptation is needed.
            # Generate a unique adapter layer name.
            adapter_name = f"{self.name}_adapter_{uuid.uuid4().hex[:6]}"
            input_tensor = layers.Dense(expected_units,   # Adapt input using a linear Dense layer.
                                        activation="linear",
                                        name=adapter_name)(input_tensor)
        return self.bypass_model(input_tensor)     # Pass the adapted tensor through the bypass model.


    @classmethod
    def load(cls, filepath, name, compile_model=False, optimizer="adam", loss="mse"):
        """
        Loads a saved subgraph model and returns it as a SubGraphNode.
        Assumes the saved model has a single input.

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
        if compile_model:
            loaded_model.compile(optimizer=optimizer, loss=loss)
        return cls(name, loaded_model)
