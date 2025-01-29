from abc import ABC, abstractmethod

class FunctionNode(ABC):
    """
    Abstract base class for function nodes in the function graph.

    Represents a node in a function graph, which is a directed acyclic graph 
    where nodes represent functions and edges represent data flow between 
    them. Each FunctionNode has a name, an input shape, and an output shape.
    It encapsulates a specific function or computation within the larger 
    network architecture.

    Eventually, this can be updated to allow (finite) cycles.
    """

    def __init__(self, name):
        """
        Initializes a FunctionNode with a given name.

        Args:
            name: The name of the function node within the graph.
        """
        self.name = name
        self.input_shape = None  # Will be set when connected in the graph
        self.output_shape = None  # Determined after building the node's internal model

    @abstractmethod
    def build(self, input_shape):
        """
        Builds the underlying TensorFlow/Keras model for the function node.

        This method defines the internal neural network structure of the node, 
        which will be executed when the node is called as a function.

        Args:
            input_shape: The shape of the input data expected by the node.
        """
        pass

    @abstractmethod
    def __call__(self, inputs):
        """
        Applies the function node to the given inputs.

        Executes the node's internal model, effectively treating the node as
        a callable function within the larger function graph.

        Args:
            inputs: The input data to the function node.

        Returns:
            The output of the function node after processing the inputs.
        """
        pass


class Identity(FunctionNode):
    def __init__(self, name):
        super().__init__(name)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # Identity: input shape = output shape

    def __call__(self, inputs):
        return inputs  # Identity: returns input unchanged

class Add(FunctionNode):
    def __init__(self, name):
        super().__init__(name)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # Add: input shape = output shape

    def __call__(self, inputs):
        return tf.add_n(inputs)  # Add all inputs together

class Multiply(FunctionNode):
    def __init__(self, name):
        super().__init__(name)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # Multiply: input shape = output shape

    def __call__(self, inputs):
        result = inputs[0]
        for x in inputs[1:]:
            result = tf.multiply(result, x)
        return result

class ReLU(FunctionNode):
    def __init__(self, name):
        super().__init__(name)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # ReLU: input shape = output shape

    def __call__(self, inputs):
        return tf.nn.relu(inputs)  # ReLU activation

class Sigmoid(FunctionNode):
    def __init__(self, name):
        super().__init__(name)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # Sigmoid: input shape = output shape

    def __call__(self, inputs):
        return tf.nn.sigmoid(inputs)  # Sigmoid activation