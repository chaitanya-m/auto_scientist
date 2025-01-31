import unittest
import tensorflow as tf
from function_graph.node import ReLU, Sigmoid, FixedReLU
from function_graph.composer import FunctionGraph, FunctionGraphComposer

class TestFunctionGraph(unittest.TestCase):
    def _test_graph_building(self, complex_graph=False):  # Helper function for both simple and complex graph building
        composer = FunctionGraphComposer()
        input_shape = (10,)  # Define input shape
        composer.set_input_shape(input_shape)  # Set input shape in composer

        input_layer = ReLU("input_layer", num_outputs=10)  # Input Layer (concrete class, num_outputs must match input_shape)
        composer.add(input_layer)

        if complex_graph:
            relu1 = ReLU("relu1", num_outputs=32)
            composer.add(relu1)
            relu2 = ReLU("relu2", num_outputs=64)
            composer.add(relu2)
            sigmoid1 = Sigmoid("sigmoid1", num_outputs=1)
            composer.add(sigmoid1)
            fixed_relu = FixedReLU("fixed_relu")
            composer.add(fixed_relu)

            composer.set_input("input_layer")
            composer.set_output("sigmoid1")

            composer.connect("input_layer", "relu1")
            composer.connect("relu1", "relu2")
            composer.connect("relu2", "sigmoid1")
            composer.connect("input_layer", "fixed_relu")

            self.assertEqual(relu1.output_shape, (None, 32))  # Correct assertion for complex graph
            self.assertEqual(relu2.output_shape, (None, 64))
            self.assertEqual(sigmoid1.output_shape, (None, 1))
            self.assertEqual(fixed_relu.output_shape, (10,)) # Fixed activation has input shape as output shape
        else:
            relu1 = ReLU("relu1", num_outputs=32)
            composer.add(relu1)
            sigmoid1 = Sigmoid("sigmoid1", num_outputs=1)
            composer.add(sigmoid1)

            composer.set_input("input_layer")
            composer.set_output("sigmoid1")

            composer.connect("input_layer", "relu1")
            composer.connect("relu1", "sigmoid1")

            self.assertEqual(relu1.output_shape, (None, 32))  # Correct assertion for simple graph
            self.assertEqual(sigmoid1.output_shape, (None, 1))

        composer.build()  # Build the graph


    def test_graph_building_simple(self):
        self._test_graph_building(complex_graph=False)

    def test_graph_building_complex(self):
        self._test_graph_building(complex_graph=True)

    def _test_training(self, complex_graph=False): # Helper function for training tests
        composer = FunctionGraphComposer()
        input_shape = (10,)  # Define input shape
        composer.set_input_shape(input_shape)  # Set input shape in composer

        input_layer = ReLU("input_layer", num_outputs=10)  # Input Layer (concrete class, num_outputs must match input_shape)
        composer.add(input_layer)

        if complex_graph:
            relu1 = ReLU("relu1", num_outputs=32)
            composer.add(relu1)
            relu2 = ReLU("relu2", num_outputs=64)
            composer.add(relu2)
            sigmoid1 = Sigmoid("sigmoid1", num_outputs=1)
            composer.add(sigmoid1)
            fixed_relu = FixedReLU("fixed_relu")
            composer.add(fixed_relu)

            composer.set_input("input_layer")
            composer.set_output("sigmoid1")

            composer.connect("input_layer", "relu1")
            composer.connect("relu1", "relu2")
            composer.connect("relu2", "sigmoid1")
            composer.connect("input_layer", "fixed_relu")
        else:
            relu1 = ReLU("relu1", num_outputs=32)
            composer.add(relu1)
            sigmoid1 = Sigmoid("sigmoid1", num_outputs=1)
            composer.add(sigmoid1)

            composer.set_input("input_layer")
            composer.set_output("sigmoid1")

            composer.connect("input_layer", "relu1")
            composer.connect("relu1", "sigmoid1")

        composer.build()  # Build the graph

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.MeanSquaredError()

        composer.compile(optimizer, loss_fn)  # CRUCIAL: Compile the graph *before* training

        input_data = tf.random.normal((100, 10))
        target_data = tf.random.normal((100, 1))

        assert hasattr(composer.graph, "optimizer"), "Optimizer not set. Did you call compile()?"
        assert hasattr(composer.graph, "loss_function"), "Loss function not set. Did you call compile()?"


        loss = composer.train(input_data, target_data, epochs=5, verbose=0)

        self.assertIsInstance(loss, tf.Tensor)
        self.assertFalse(tf.math.is_nan(loss))


    def test_training_simple(self):
        self._test_training(complex_graph=False)

    def test_training_complex(self):
        self._test_training(complex_graph=True)