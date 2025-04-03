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
    def setUp(self):
        self.agent = SimpleMCTSAgent()
    
    def test_initial_state(self):
        state = self.agent.get_initial_state()
        self.assertIn("composer", state)
        self.assertIn("graph_actions", state)
        self.assertIn("performance", state)
        self.assertIn("target_mse", state)
        self.assertIsInstance(state["composer"], GraphComposer)
        # Check that the model built has an output dimension equal to latent_dim.
        state["composer"].build()
        output_dim = state["composer"].keras_model.output.shape[-1]
        self.assertEqual(output_dim, self.agent.latent_dim)
    
    def test_get_available_actions(self):
        state = self.agent.get_initial_state()
        actions = self.agent.get_available_actions(state)
        # Expect at least two actions: add_neuron and delete_repository_entry.
        self.assertIn("add_neuron", actions)
        self.assertIn("delete_repository_entry", actions)
    
    def test_apply_add_neuron_action(self):
        state = self.agent.get_initial_state()
        initial_actions_length = len(state["graph_actions"])
        new_state = self.agent.apply_action(state, "add_neuron")
        self.assertEqual(len(new_state["graph_actions"]), initial_actions_length + 1)
        # Check that the new graph can build a valid model.
        new_state["composer"].build()
        self.assertIsNotNone(new_state["composer"].keras_model)
    
    def test_apply_delete_repository_action(self):
        # For delete_repository_entry to work, add something to repository first.
        self.agent.repository.append("dummy_entry")
        state = self.agent.get_initial_state()
        initial_repo_length = len(self.agent.repository)
        new_state = self.agent.apply_action(state, "delete_repository_entry")
        self.assertEqual(len(self.agent.repository), initial_repo_length - 1)


if __name__ == '__main__':
    unittest.main()
