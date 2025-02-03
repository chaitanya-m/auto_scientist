import unittest
import tensorflow as tf
from function_graph.node import (
    Trainable,
    TrainableNN,
    Sigmoid,
    ReLU
)


class TestFunctionNodes(unittest.TestCase):
    """
    Test suite for the FunctionNode classes, including trainable and fixed activations.
    """

    def _test_node_creation(self, node_class, expected_name):
        """
        Tests the creation of a FunctionNode, verifying name and initial shape.

        Args:
            node_class: The class of the node to create (e.g., TrainableNN).
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
            ReLU("my_relu"),
            (10, 5),
            (10, 1),
            lambda output: self.assertTrue(tf.reduce_all(output >= 0))
        )
        self._test_call_with_multiple_inputs(ReLU("my_relu"), (10, 2), (10, 3), 1)

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
             - Assert that the output shape is (10, 1), as expected for a Sigmoid node with 1 output features.
             - Call the provided lambda function `lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and self.assertTrue(tf.reduce_all(output <= 1)))` to perform extra checks.  This lambda function asserts that all elements in the output tensor are between 0 and 1 (inclusive), which is a defining characteristic of the Sigmoid activation function.

        2. **Multiple Input Call:**
           - Calls the `_test_call_with_multiple_inputs` helper method to test the Sigmoid node's behavior with multiple input tensors.
           - The inputs are (10, 2) and (10, 3) which will be concatenated to (10,5)
           - The expected output features are 2 because that is how the Sigmoid was initialized.
           - This helper method concatenates the input tensors, builds the node, calls the node, and checks the output shape. It also re-asserts the Sigmoid characteristic (0 <= output <= 1).
        """
        self._test_activation_build_and_call(
            Sigmoid("my_sigmoid"),
            (10, 5),
            (10, 1),
            lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and
                            self.assertTrue(tf.reduce_all(output <= 1)))
        )
        self._test_call_with_multiple_inputs(Sigmoid("my_sigmoid"), (10, 2), (10, 3), 1)


    def test_trainable_nn_creation(self):
        """Tests the creation of TrainableNN nodes."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2, activation="sigmoid")
        ])
        node = TrainableNN("trainable_nn", model)
        self.assertEqual(node.name, "trainable_nn")
        self.assertIsInstance(node, Trainable)


    def test_trainable_nn_build_and_call(self):
        """Tests the build and forward pass of TrainableNN."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2, activation="sigmoid")
        ])
        node = TrainableNN("trainable_nn", model)

        input_shape = (10, 5)
        node.build(input_shape)
        self.assertEqual(node.input_shape, input_shape)
        self.assertEqual(node.output_shape, model.output_shape)

        inputs = tf.random.normal(input_shape)
        output = node(inputs)
        self.assertEqual(output.shape[1:], node.output_shape[1:]) 
        # Batch size for keras model is None for genericity; only compares number of features




if __name__ == "__main__":
    unittest.main()