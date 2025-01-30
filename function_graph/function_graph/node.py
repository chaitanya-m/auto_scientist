from abc import ABC, abstractmethod
import tensorflow as tf

class FunctionNode(ABC):
    def __init__(self, name: str):
        self.name = name
        self.input_shape = None
        self.output_shape = None

    @abstractmethod
    def build(self, input_shape):
        pass

    @abstractmethod
    def __call__(self, inputs):
        pass

class Trainable(ABC):
    @abstractmethod
    def train(self, inputs, targets, optimizer, loss_function, epochs=1, verbose=0):
        pass

class TrainableNode(FunctionNode, Trainable):
    def add_weight(self, name, shape):
        return tf.Variable(tf.random.normal(shape), name=name)

    def add_bias(self, name, shape):
        return tf.Variable(tf.zeros(shape), name=name)

    def train(self, inputs, targets, optimizer, loss_function, epochs=1, verbose=0):
        inputs = tf.cast(inputs, tf.float32)
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = loss_function(targets, outputs)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @property
    def trainable_variables(self):
        return self._trainable_variables()

    def _trainable_variables(self):
        trainable_vars = []
        for var in self.__dict__.values():
            if isinstance(var, tf.Variable):
                trainable_vars.append(var)
            elif isinstance(var, TrainableNode):
                trainable_vars.extend(var.trainable_variables)
        return trainable_vars


class FixedActivation(FunctionNode):
    def __init__(self, name: str, activation_function=tf.nn.sigmoid):
        super().__init__(name)
        self.activation_function = activation_function

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def __call__(self, inputs):
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        inputs = tf.cast(inputs, tf.float32)
        return self.activation_function(inputs)

class TrainableActivation(TrainableNode):
    def __init__(self, name: str, activation_function=tf.nn.sigmoid, num_outputs=1):
        super().__init__(name)
        self.activation_function = activation_function
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[-1]
        self.W = self.add_weight("W", shape=(input_dim, self.num_outputs))
        self.b = self.add_bias("b", shape=(self.num_outputs,))

        input_rank = tf.TensorShape(input_shape).rank
        if input_rank == 1:
            self.output_shape = (self.num_outputs,)
        elif input_rank == 2:
            self.output_shape = (None, self.num_outputs)
        else:
            raise ValueError("Input shape must have rank 1 or 2.")

    def __call__(self, inputs):
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        inputs = tf.cast(inputs, tf.float32)
        z = tf.matmul(inputs, self.W) + self.b
        return self.activation_function(z)

class FixedSigmoid(FixedActivation):
    def __init__(self, name: str):
        super().__init__(name, activation_function=tf.nn.sigmoid)

class FixedReLU(FixedActivation):
    def __init__(self, name: str):
        super().__init__(name, activation_function=tf.nn.relu)


class ReLU(TrainableNode):
    def __init__(self, name, num_outputs):
        super().__init__(name)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.input_shape = input_shape
        input_rank = tf.TensorShape(input_shape).rank
        if input_rank == 1:
            self.output_shape = (self.num_outputs,)
        elif input_rank == 2:
            self.output_shape = (None, self.num_outputs)
        else:
            raise ValueError("Input shape must have rank 1 or 2.")

        self.W = self.add_weight("W", shape=(input_shape[-1], self.num_outputs)) # Corrected shape
        self.b = self.add_bias("b", shape=(self.num_outputs,)) # Corrected shape

    def __call__(self, inputs):
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        z = tf.matmul(inputs, self.W) + self.b
        return tf.maximum(0., z)


class Sigmoid(TrainableNode):
    def __init__(self, name, num_outputs):
        super().__init__(name)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.input_shape = input_shape
        input_rank = tf.TensorShape(input_shape).rank
        if input_rank == 1:
            self.output_shape = (self.num_outputs,)
        elif input_rank == 2:
            self.output_shape = (None, self.num_outputs)
        else:
            raise ValueError("Input shape must have rank 1 or 2.")

        self.W = self.add_weight("W", shape=(input_shape[-1], self.num_outputs))  # Corrected shape
        self.b = self.add_bias("b", shape=(self.num_outputs,))  # Corrected shape

    def __call__(self, inputs):
        if isinstance(inputs, list):
            batch_sizes = [tf.shape(x)[0] for x in inputs]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("Input tensors must have consistent batch sizes for concatenation.")
            inputs = tf.concat(inputs, axis=-1)
        z = tf.matmul(inputs, self.W) + self.b
        return tf.sigmoid(z)