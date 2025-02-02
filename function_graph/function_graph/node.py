from abc import ABC, abstractmethod
from typing import Union, List
import tensorflow as tf

class FunctionNode(ABC):
    """
    Abstract base class representing a node in a function graph.

    A FunctionNode represents a computation step in a directed acyclic graph (DAG) of computations (finite cycles in future?).

    Nodes can either be trainable or fixed, and they transform their inputs according to suit the shape of the 
    nodes that feed into them. That is, each node has an input shape and an output shape, which are defined when the node is 
    connected in the graph.
    
    Nodes contain functions. For a start, we will use the neural net paradigm mainly due to accessibility.
    
    A function in a node may be fixed or learnable (allowing exploration of function space). In the neural net context, we mean
    neural nets at nodes. We may have layers, multiple layers, activation functions, etc. that operate on the incoming tensor.

    Attributes:
        name: The name of the function node.
        input_shape: The shape of the input data expected by the node. It is set during the build phase.
        output_shape: The shape of the output data produced by the node after computation.
    """

    def __init__(self, name: str):
        """
        Initializes a FunctionNode with the given name.

        Args:
            name: The name of the function node within the graph.
        """
        self.name = name
        self.input_shape = None  # Will be set during the build phase - can this be better streamlined?
        self.output_shape = None  # Determined after building the node's internal model

    @abstractmethod
    def build(self, input_shape):
        """
        Builds the underlying model for the function node based on the input shape.

        This method should define the structure of the node (e.g., weights, biases, etc. for a learnable node).

        Args:
            input_shape: The shape of the input data expected by the node. This is typically 
                          passed when the node is connected to the graph.
        """
        pass

    @abstractmethod
    def __call__(self, inputs):
        """
        Computes the output of the node based on the given inputs.

        Args:
            inputs: The input data to the function node. Can be a single tensor or a list of tensors
                    if the node has multiple inputs.

        Returns:
            The output of the function node after processing the inputs, i.e. the function has been called.
        """
        pass

class Trainable(ABC):
    """
    Abstract base class for trainable nodes that have learnable parameters.
    
    This class defines the interface for function nodes that need to train their parameters
    (e.g., weights, biases) using a training process such as backpropagation.
    """

    @abstractmethod
    def train(self, inputs, targets, optimizer, loss_function, epochs=1, verbose=0):
        """
        Trains the node's learnable parameters using the provided inputs and targets.

        Args:
            inputs: The input data used to train the node.
            targets: The target data used for comparison during training.
            optimizer: The optimizer used to update the parameters of the node.
            loss_function: The loss function used to compute the error between predicted and target outputs.
            epochs: The number of training iterations (epochs).
            verbose: The level of verbosity for training logs (0 = no logs, 1 = training progress).

        Returns:
            The final loss value after training.
        """
        pass

class TrainableNode(FunctionNode, Trainable):
    """
    A concrete class representing a trainable node that contains both function node behavior
    and training functionality (learnable weights and biases).

    This class provides functionality for adding weights and biases, as well as a method to 
    train the node's parameters using a training loop (e.g., backpropagation).

    """

    def add_weight(self, name, shape):
        """Creates and initializes a weight variable with the given name and shape.

        Args:
            name: The name of the weight variable (e.g., "weights_layer1").
            shape: A tuple or list specifying the dimensions of the weight matrix 
                (e.g., (input_dim, output_dim)).

        Returns:
            A TensorFlow variable representing the weight matrix.
        """
        return tf.Variable(tf.random.normal(shape), name=name)

    def add_bias(self, name, shape):
        """Creates and initializes a weight variable with the given name and shape.

        Args:
            name: The name of the weight variable (e.g., "weights_layer1").
            shape: A tuple or list specifying the dimensions of the weight matrix 
                (e.g., (input_dim, output_dim)).

        Returns:
            A TensorFlow variable representing the weight matrix.
        """
        return tf.Variable(tf.zeros(shape), name=name)

    def train(self, inputs, targets, optimizer, loss_function, epochs=1, verbose=0):
        """
        Trains the node using the provided optimizer, loss function, and data.

        Args:
            inputs: The input data for training.
            targets: The target values for training.
            optimizer: The optimizer used for training the model.
            loss_function: The loss function used for error calculation.
            epochs: The number of training epochs.
            verbose: Verbosity level for progress logs.

        Returns:
            The final loss after training.
        """
        inputs = tf.cast(inputs, tf.float32)
        with tf.GradientTape() as tape:
            outputs = self(inputs) # perform forward pass
            loss = loss_function(targets, outputs) # compute loss
        gradients = tape.gradient(loss, self.trainable_variables) # calculate loss gradients for all trainable variables
        optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # apply gradients, to update all trainable variables 
        return loss

    @property
    def trainable_variables(self):
        """
        Returns a list of all trainable variables (weights and biases) for the node.

        This includes any variables (e.g., weights, biases) that need to be updated during training.
        """
        return self._trainable_variables()

    def _trainable_variables(self):
        """
        Helper method to collect all trainable variables from the node and its sub-nodes.

        Returns:
            A list of all TensorFlow variables that are trainable within the node.
        """
        trainable_vars = []
        for var in self.__dict__.values():
            if isinstance(var, tf.Variable):
                trainable_vars.append(var)
            elif isinstance(var, TrainableNode):
                trainable_vars.extend(var.trainable_variables)
        return trainable_vars

class FixedActivation(FunctionNode):
    """
    Base class for fixed (non-trainable) activation functions.

    This class implements fixed activations that do not have learnable parameters. 
    
    The output shape is the same as the input shape since activations typically don't change
    the dimensions of the data.

    Attributes:
        activation_function: The TensorFlow activation function to be applied to the inputs.
    """

    def __init__(self, name: str, activation_function=tf.nn.sigmoid):
        """
        Initializes a FixedActivation.

        Args:
            name: The name of the activation node.
            activation_function: The TensorFlow activation function (default is tf.nn.sigmoid).
        """
        super().__init__(name)
        self.activation_function = activation_function

    def build(self, input_shape):
        """
        Sets the input and output shapes for the activation function.

        The output shape is the same as the input shape, as activation functions typically 
        do not change the dimensionality of the data.

        Args:
            input_shape: The shape of the input data to the activation function.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

    def __call__(self, inputs):
        """
        Applies the activation function to the given inputs.

        Args:
            inputs: The input data (single tensor or a list of tensors).

        Returns:
            The output after applying the activation function.
        """
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        inputs = tf.cast(inputs, tf.float32)
        return self.activation_function(inputs)


class TrainableNN(TrainableNode):
    """Represents a trainable neural network node in a computational graph.

    This node wraps a Keras model, allowing it to be used as a unit
    in a larger computational graph.
    """
    def __init__(self, name: str, model: tf.keras.Model):
        """Initializes the TrainableNN."""
        super().__init__(name)
        self.model = model
        self._is_built = False

    def build(self, input_shape):
        """Builds the Keras model (only once) and sets input/output shapes."""
        if self._is_built and self.input_shape != input_shape:
            raise ValueError(f"TrainableNN '{self.name}' was already built with a different input shape: {self.input_shape}, got {input_shape}.")
        if not self._is_built:
            self.input_shape = input_shape
            self.model.build(input_shape)
            self.output_shape = self.model.output_shape
            self._is_built = True


    def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]):
        """Performs a forward pass while ensuring correct input format."""
        if isinstance(inputs, list):
            inputs = [tf.convert_to_tensor(i) for i in inputs]
        return self.model(inputs)  # Forward pass

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def save_weights(self, filepath: str):
        """Saves the model weights."""
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str):
        """Loads weights into the model."""
        self.model.load_weights(filepath)


class FixedSigmoid(FixedActivation):
    """
    A fixed (non-trainable) Sigmoid activation function.

    This class represents a sigmoid activation function that does not have learnable parameters.
    It is used to apply a sigmoid transformation to the input data in a non-trainable manner, meaning
    that no weights or biases are involved. The output shape is the same as the input shape, and the 
    sigmoid function is applied element-wise to the input.

    In neural networks, sigmoid activations are often used in binary classification problems 
    or to introduce non-linearity in the model.

    Inherits from:
        FixedActivation: Provides the core structure for applying fixed activation functions.

    """

    def __init__(self, name: str):
        """
        Initializes the FixedSigmoid activation function.

        Args:
            name: The name of the Sigmoid node in the graph.
        """
        super().__init__(name, activation_function=tf.nn.sigmoid)

class FixedReLU(FixedActivation):
    """
    A fixed (non-trainable) ReLU activation function.

    This class represents a ReLU (Rectified Linear Unit) activation function that does not have learnable 
    parameters. ReLU is a very common activation function used in deep learning models because it 
    introduces non-linearity and is computationally efficient.

    The ReLU activation is applied element-wise to the input, where all negative values are set to zero 
    and positive values are left unchanged.

    Inherits from:
        FixedActivation: Provides the core structure for applying fixed activation functions.

    """

    def __init__(self, name: str):
        """
        Initializes the FixedReLU activation function.

        Args:
            name: The name of the ReLU node in the graph.
        """
        super().__init__(name, activation_function=tf.nn.relu)

class ReLU(TrainableNode):
    """
    A trainable ReLU activation function.

    This class represents a ReLU activation function that has learnable parameters. 
    This is a custom activation that includes trainable weights and biases. 
    Typically, a ReLU function does not include parameters, but in this case, we allow 
    the possibility of learning weights and biases that are then added to the input 
    before applying the ReLU function.

    ReLU applies the following transformation element-wise to the input:
        - If input > 0: return input
        - If input <= 0: return 0

    Inherits from:
        TrainableNode: Allows the node to have learnable parameters and be part of a training process.

    Attributes:
        num_outputs: The number of output neurons from the ReLU function (determines the size of the output).

    """

    def __init__(self, name: str, num_outputs: int):
        """
        Initializes the ReLU activation function with learnable parameters.

        Args:
            name: The name of the ReLU node in the graph.
            num_outputs: The number of output neurons from this activation function.
        """
        super().__init__(name)
        self.num_outputs = num_outputs

    # def build(self, input_shape):
    #     """
    #     Builds the ReLU node by initializing weights and biases.

    #     The weights and biases are initialized randomly, and the output shape is set based on 
    #     the number of output neurons. The weights are initialized as random values, and biases are initialized as zeros.

    #     Args:
    #         input_shape: The shape of the input data expected by the ReLU node.
    #     """
    #     self.input_shape = input_shape
    #     input_rank = tf.TensorShape(input_shape).rank
    #     if input_rank == 1:
    #         self.output_shape = (self.num_outputs,)
    #     elif input_rank == 2:
    #         self.output_shape = (None, self.num_outputs)
    #     else:
    #         raise ValueError("Input shape must have rank 1 or 2.")

    #     self.W = self.add_weight("W", shape=(input_shape[-1], self.num_outputs))  # Initialize weights
    #     self.b = self.add_bias("b", shape=(self.num_outputs,))  # Initialize biases

    def build(self, input_shape):
        self.input_shape = input_shape # usually (batch_size, num_features)
        input_dim = input_shape[-1] # number of features
        self.W = self.add_weight("W", shape=(input_dim, self.num_outputs))
        self.b = self.add_bias("b", shape=(self.num_outputs,))

        self.output_shape = (None, self.num_outputs)  # Consistent rank 2 output

    def __call__(self, inputs):
        """
        Performs a forward pass through the ReLU activation layer.

        This method applies a linear transformation (matrix multiplication with weights and 
        addition of a bias) followed by the ReLU activation function element-wise.  Handles 
        input concatenation if a list of tensors is provided.

        Args:
            inputs: Input tensor(s) to the ReLU layer. Can be a single tensor or a list of 
                    tensors. If a list is provided, the tensors are concatenated along the 
                    last axis (axis=-1) after verifying consistent batch sizes.

        Returns:
            The output tensor after the linear transformation and ReLU activation.
            
        Raises:
            ValueError: If input tensors provided as a list do not have consistent 
                            batch sizes.
        """

        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        z = tf.matmul(inputs, self.W) + self.b  # Linear transformation - multiply by weight, add bias
        return tf.maximum(0., z)  # Apply ReLU elementwise to the tensor: returns the max between 0 and the input

class Sigmoid(TrainableNode):
    """
    A trainable Sigmoid activation function.

    This class represents a Sigmoid activation function that has learnable weights and biases. 
    While standard Sigmoid functions don't have parameters, this class allows the Sigmoid 
    to include trainable weights and biases for advanced architectures that may require such 
    flexibility.

    The Sigmoid function applies the following transformation to the input:
        Sigmoid(x) = 1 / (1 + exp(-x))

    This transformation squashes the output between 0 and 1.

    Inherits from:
        TrainableNode: Enables the Sigmoid function to have learnable parameters and to be part of a training process.

    Attributes:
        num_outputs: The number of output neurons from the Sigmoid activation.

    """

    def __init__(self, name: str, num_outputs: int):
        """
        Initializes the Sigmoid activation function with learnable parameters.

        Args:
            name: The name of the Sigmoid node in the graph.
            num_outputs: The number of output neurons from this activation function.
        """
        super().__init__(name)
        self.num_outputs = num_outputs

    # def build(self, input_shape):
    #     """
    #     Builds the Sigmoid node by initializing weights and biases.

    #     Similar to the ReLU node, weights and biases are initialized randomly, and the 
    #     output shape is determined by the number of output neurons.

    #     Args:
    #         input_shape: The shape of the input data expected by the Sigmoid node.
    #     """
    #     self.input_shape = input_shape
    #     input_rank = tf.TensorShape(input_shape).rank
    #     if input_rank == 1:
    #         self.output_shape = (self.num_outputs,)
    #     elif input_rank == 2:
    #         self.output_shape = (None, self.num_outputs)
    #     else:
    #         raise ValueError("Input shape must have rank 1 or 2.")

    #     self.W = self.add_weight("W", shape=(input_shape[-1], self.num_outputs))  # Initialize weights
    #     self.b = self.add_bias("b", shape=(self.num_outputs,))  # Initialize biases

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[-1]
        self.W = self.add_weight("W", shape=(input_dim, self.num_outputs))
        self.b = self.add_bias("b", shape=(self.num_outputs,))

        self.output_shape = (None, self.num_outputs)  # Consistent rank 2 output

    def __call__(self, inputs):
        """
        Applies the Sigmoid function after computing the weighted sum of inputs.

        First, the input is passed through a linear transformation (matrix multiplication 
        with weights and bias), and then the Sigmoid function is applied to squash the 
        output between 0 and 1.

        Args:
            inputs: The input data to the Sigmoid activation function, which can be a single tensor 
                    or a list of tensors. If a list is provided, they will be concatenated.

        Returns:
            The output after applying the Sigmoid activation function.
        """
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        z = tf.matmul(inputs, self.W) + self.b  # Linear transformation
        return tf.sigmoid(z)  # Apply Sigmoid activation: squashes between 0 and 1
