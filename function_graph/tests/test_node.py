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

    def _test_node_creation(self, node_class, expected_name):
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsNone(node.input_shape)
        self.assertIsNone(node.output_shape)

    def _test_trainable_creation(self, node_class, expected_name):
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsInstance(node, Trainable)

    def _test_fixed_activation_creation(self, node_class, expected_name):
        node = node_class(expected_name)
        self.assertEqual(node.name, expected_name)
        self.assertIsInstance(node, FixedActivation)

    def _test_activation_build_and_call(self, activation_node, input_shape, expected_output_shape, extra_checks=None):
        activation_node.build(input_shape)
        inputs = tf.random.normal(input_shape)
        output = activation_node(inputs)
        self.assertEqual(output.shape, expected_output_shape)

        if extra_checks:
            extra_checks(output)

    def _test_call_with_multiple_inputs(self, node, input_shape1, input_shape2, expected_output_features):
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
        self._test_trainable_creation(TrainableActivation, "trainable_activation") 

    def test_fixed_activation_creation(self):
        self._test_fixed_activation_creation(FixedActivation, "fixed_activation") 

    def test_relu_build_and_call(self):
        self._test_activation_build_and_call(
            ReLU("my_relu", num_outputs=1),
            (10, 5),
            (10, 1),
            lambda output: self.assertTrue(tf.reduce_all(output >= 0))
        )
        self._test_call_with_multiple_inputs(ReLU("my_relu", num_outputs=1), (10, 2), (10, 3), 1)  # Corrected: 1 output feature


    def test_sigmoid_build_and_call(self):
        self._test_activation_build_and_call(
            Sigmoid("my_sigmoid", num_outputs=2),
            (10, 5),
            (10, 2),
            lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and
                            self.assertTrue(tf.reduce_all(output <= 1)))
        )
        self._test_call_with_multiple_inputs(Sigmoid("my_sigmoid", num_outputs=2), (10, 2), (10, 3), 2)  # Corrected: 2 output features

    def test_fixed_sigmoid_build_and_call(self):
        self._test_activation_build_and_call(
            FixedSigmoid("fixed_sigmoid"),
            (10, 5),
            (10, 5),
            lambda output: (self.assertTrue(tf.reduce_all(output >= 0)) and
                            self.assertTrue(tf.reduce_all(output <= 1)))
        )
        self._test_call_with_multiple_inputs(FixedSigmoid("fixed_sigmoid"), (10, 2), (10, 3), 5)  # Corrected: 5 output features

    def test_fixed_relu_build_and_call(self):
        self._test_activation_build_and_call(
            FixedReLU("fixed_relu"),
            (10, 5),
            (10, 5),
            lambda output: self.assertTrue(tf.reduce_all(output >= 0))
        )
        self._test_call_with_multiple_inputs(FixedReLU("fixed_relu"), (10, 2), (10, 3), 5)  # Corrected: 5 output features

    def test_trainable_activation_train(self):
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