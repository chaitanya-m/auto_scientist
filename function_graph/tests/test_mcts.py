import unittest
import tensorflow as tf

from utils.nn import create_minimal_graphmodel
from agents.mcts import SimpleMCTSAgent
from graph.composer import GraphComposer

class TestMinimalGraphModel(unittest.TestCase):
    def test_single_neuron_case(self):
        # Test default behavior with a single neuron output.
        input_shape = (3,)
        # When output_units is not specified, it defaults to 1.
        composer, model = create_minimal_graphmodel(input_shape)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsNotNone(model.input)
        self.assertIsNotNone(model.output)
        
        # Verify that the model's output has one unit.
        output_shape = model.output.shape
        self.assertEqual(output_shape[-1], 1)
        
        # Verify that the composer has the expected node names.
        self.assertEqual(composer.input_node_name, "input")
        self.assertIn("output", composer.output_node_names)

    def test_encoder_case(self):
        # Test the encoder case where output_units > 1.
        input_shape = (3,)
        output_units = 4  # For example, a latent space of 4 units.
        activation = "relu"
        composer, model = create_minimal_graphmodel(input_shape, output_units=output_units, activation=activation)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsNotNone(model.input)
        self.assertIsNotNone(model.output)
        
        # Verify that the model's output has the correct number of units.
        output_shape = model.output.shape
        self.assertEqual(output_shape[-1], output_units)
        
        # Verify that the composer has the expected node names.
        self.assertEqual(composer.input_node_name, "input")
        self.assertIn("output", composer.output_node_names)

class TestSimpleMCTSAgent(unittest.TestCase):
    def test_initial_state(self):
        agent = SimpleMCTSAgent()
        state = agent.get_initial_state()
        
        # Verify that the state includes all required keys.
        self.assertIn("composer", state)
        self.assertIn("graph_actions", state)
        self.assertIn("performance", state)
        self.assertIn("target_mse", state)
        
        # Verify that the composer is an instance of GraphComposer.
        self.assertIsInstance(state["composer"], GraphComposer)
        self.assertEqual(state["composer"].input_node_name, "input")
        self.assertIn("output", state["composer"].output_node_names)
        
        # Ensure that the model output dimension equals the agent's latent dimension.
        model_output_dim = state["composer"].keras_model.output.shape[-1]
        self.assertEqual(model_output_dim, agent.latent_dim)

if __name__ == '__main__':
    unittest.main()
