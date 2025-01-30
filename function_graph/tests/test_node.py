import unittest
import tensorflow as tf
from function_graph.node import (
    Trainable,
    TrainableActivation,
    FixedActivation,
    FixedSigmoid,
    FixedReLU,
    Sigmoid,
    ReLU
)
import numpy as np

class TestFunctionNodes(unittest.TestCase):
    """
    Test suite for the FunctionNode classes, including trainable and fixed activations.
    """

    def _test_node_creation(self, node_class, expected_name):
        """
        Tests the creation of a FunctionNode, verifying name and initial shape.

        Args:
            node_class: The class of the node to create (e.g., TrainableActivation).
            expected_name: The expected name of the node.
        """
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsNone(node.input_shape)
        self.assertIsNone(node.output_shape)

    def _test_trainable_creation(self, node_class, expected_name):
        """
        Tests the creation of a trainable node, verifying name and Trainable interface.

        Args:
            node_class: The class of the trainable node to create.
            expected_name: The expected name of the node.
        """
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsInstance(node, Trainable)

    def _test_fixed_activation_creation(self, node_class, expected_name):
        """
        Tests the creation of a fixed activation node, verifying name and FixedActivation interface.

        Args:
            node_class: The class of the fixed activation node to create.
            expected_name: The expected name of the node.
        """
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsInstance(node, FixedActivation)

    def _test_activation_build_and_call(self, activation_node, input_shape, expected_output_shape, extra_checks=None):
        """
        Tests the build and call methods of an activation node.

        Args:
            activation_node: The activation node instance.
            input_shape: The input shape for the build method.
            expected_output_shape: The expected output shape after the build and call methods.
            extra_checks: An optional function to perform additional checks on the output.
        """
        activation_node.build(input_shape)
        inputs = tf.random.normal(input_shape)
        output = activation_node(inputs)
        self.assertEqual(output.shape, expected_output_shape)

        if extra_checks:
            extra_checks(output)

    def _test_call_with_multiple_inputs(self, node, input_shape1, input_shape2, expected_output_features):
        """
        Tests the call method of a node with multiple inputs.

        Args:
            node: The node instance.
            input_shape1: The shape of the first input.
            input_shape2: The shape of the second input.
            expected_output_features: The expected number of output features after concatenation and processing.
        """
        inputs1 = tf.random.normal(input_shape1)
        inputs2 = tf.random.normal(input_shape2)
        concatenated_inputs = tf.concat([inputs1, inputs2], axis=-1)
        node.build(concatenated_inputs.shape)  # CRUCIAL: Build with concatenated shape
        output = node(concatenated_inputs)

        self.assertEqual(output.shape, (input_shape1[0], expected_output_features))

        if isinstance(node, ReLU):
            self.assertTrue(tf.reduce_all(output >= 0))
        elif isinstance(node, Sigmoid):
            self.assertTrue(tf.reduce_all(output >= 0))
            self.assertTrue(tf.reduce_all(output <= 1))   

    def test_trainable_creation(self):
        """Tests the creation of TrainableActivation nodes."""
        self._test_trainable_creation(TrainableActivation, "trainable_activation") 

    def test_fixed_activation_creation(self):
        """Tests the creation of FixedActivation nodes."""
        self._test_fixed_activation_creation(FixedActivation, "fixed_activation") 

    def test_relu_build_and_call(self):
        """
        Tests the build and call methods of a ReLU (Rectified Linear Unit) node.

        This test verifies the correct initialization and functionality of a ReLU node.  It performs the following steps:

        1. **Node Creation and Build:**
           - Creates a ReLU node instance named "my_relu" with 1 output feature.
           - Calls the `_test_activation_build_and_call` helper method to:
             - Build the ReLU node, initializing its weights and biases based on the input shape (10, 5). This sets up the internal structure of the node for processing data.
             - Generate random input data with shape (10, 5). This simulates real-world input to the node.
             - Call the ReLU node with the input data. This performs the forward pass calculation: `output = ReLU(input)`.
             - Assert that the output shape is (10, 1), as expected for a ReLU node with 1 output feature.
             - Call the provided lambda function `lambda output: self.assertTrue(tf.reduce_all(output >= 0))` to perform an extra check.  This lambda function asserts that all elements in the output tensor are greater than or equal to 0, which is a defining characteristic of the ReLU activation function.

        2. **Multiple Input Call:**
           - Calls the `_test_call_with_multiple_inputs` helper method to test the ReLU node's behavior with multiple input tensors. This simulates a scenario where the input to the node is composed of multiple parts.
           - The inputs are (10, 2) and (10, 3) which will be concatenated to (10,5)
           - The expected output features are 1 because that is how the ReLU was initialized.
           - This helper method concatenates the input tensors along the last axis, builds the node based on the concatenated input shape, calls the node with the concatenated input, and checks if the output shape is correct.  It also re-asserts the ReLU characteristic (output >= 0).
        """
        self._test_activation_build_and_call(
            ReLU("my_relu", num_outputs=1),
            (10, 5),
            (10, 1),
            lambda output: self.assertTrue(tf.reduce_all(output >= 0))
        )
        self._test_call_with_multiple_inputs(ReLU("my_relu", num_outputs=1), (10, 2), (10, 3), 1)

    def test_sigmoid_build_and_call(self):
        """
        Tests the build and call methods of a Sigmoid node.

        This test is similar to the `test_relu_build_and_call` test, but it focuses on the Sigmoid activation function.  It performs the following steps:

        1. **Node Creation and Build:**
           - Creates a Sigmoid node instance named "my_sigmoid" with 2 output features.
           - Calls the `_test_activation_build_and_call` helper method to:
             - Build the Sigmoid node, initializing its weights and biases based on the input shape (10, 5).
             - Generate random input data with shape (10, 5).
             - Call the Sigmoid node with the input data.
             - Assert that the output shape is (10, 2), as expected for a Sigmoid node with 2 output features.
             - Call the provided lambda function `lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and self.assertTrue(tf.reduce_all(output <= 1)))` to perform extra checks.  This lambda function asserts that all elements in the output tensor are between 0 and 1 (inclusive), which is a defining characteristic of the Sigmoid activation function.

        2. **Multiple Input Call:**
           - Calls the `_test_call_with_multiple_inputs` helper method to test the Sigmoid node's behavior with multiple input tensors.
           - The inputs are (10, 2) and (10, 3) which will be concatenated to (10,5)
           - The expected output features are 2 because that is how the Sigmoid was initialized.
           - This helper method concatenates the input tensors, builds the node, calls the node, and checks the output shape. It also re-asserts the Sigmoid characteristic (0 <= output <= 1).
        """
        self._test_activation_build_and_call(
            Sigmoid("my_sigmoid", num_outputs=2),
            (10, 5),
            (10, 2),
            lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and
                            self.assertTrue(tf.reduce_all(output <= 1)))
        )
        self._test_call_with_multiple_inputs(Sigmoid("my_sigmoid", num_outputs=2), (10, 2), (10, 3), 2)

    def test_fixed_sigmoid_build_and_call(self):
        """Tests the build and call methods of a FixedSigmoid node."""
        self._test_activation_build_and_call(
            FixedSigmoid("fixed_sigmoid"),
            (10, 5),
            (10, 5),
            lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and
                            self.assertTrue(tf.reduce_all(output <= 1)))
        )
        self._test_call_with_multiple_inputs(FixedSigmoid("fixed_sigmoid"), (10, 2), (10, 3), 5)

    def test_fixed_relu_build_and_call(self):
        """Tests the build and call methods of a FixedReLU node."""
        self._test_activation_build_and_call(
            FixedReLU("fixed_relu"),
            (10, 5),
            (10, 5),
            lambda output: self.assertTrue(tf.reduce_all(output >= 0))
        )
        self._test_call_with_multiple_inputs(FixedReLU("fixed_relu"), (10, 2), (10, 3), 5)

    def test_trainable_activation_train(self):
        """
        Tests the train method of a TrainableActivation node, verifying weight and bias updates.

        This test performs the following steps:
        1. Initializes a TrainableActivation node with a specified number of outputs.
        2. Builds the node with a given input shape, initializing the weights (W) and biases (b).
        3. Generates random input data and target data.  The target data's shape is crucial; it must match the output shape of the TrainableActivation node.
        4. Initializes an Adam optimizer with a learning rate.  Setting the learning rate is important for the test to be meaningful.
        5. Defines a MeanSquaredError loss function.
        6. Stores the initial values of the weights (W) and biases (b) before training.
        7. Calls the `train` method of the TrainableActivation node, performing one training step. This calculates the loss, computes gradients, and updates the weights and biases using the optimizer.
        8. Asserts that the returned loss is a TensorFlow tensor.
        9. Retrieves the values of the weights (W) and biases (b) *after* training.
        10. Asserts that the weights and biases have been updated after the training step.  This verifies that the `train` method correctly calculates and applies gradients.

        The test uses a fixed random seed (`tf.random.set_seed(42)`) to ensure consistent weight initialization and test reproducibility.  This means that the initial weights and biases will be the same every time the test is run, making the test deterministic.
        """

        tf.random.set_seed(42)
        activation = TrainableActivation("trainable_sigmoid", num_outputs=2)
        input_shape = (10, 5)
        activation.build(input_shape)

        inputs = tf.random.normal(input_shape)
        targets = tf.random.normal((10, 2))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        loss_fn = tf.keras.losses.MeanSquaredError()

        W_before_train = activation.W.numpy()
        b_before_train = activation.b.numpy()

        loss = activation.train(inputs, targets, optimizer, loss_fn)

        self.assertIsInstance(loss, tf.Tensor)

        W_after_train = activation.W.numpy()
        b_after_train = activation.b.numpy()

        self.assertFalse(np.array_equal(W_before_train, W_after_train), "Weights should be updated")
        self.assertFalse(np.array_equal(b_before_train, b_after_train), "Biases should be updated")

if __name__ == "__main__":
    unittest.main()