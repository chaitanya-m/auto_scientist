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
        # Ensure the built model has an output dimension equal to the latent_dim.
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
        new_state["composer"].build()
        self.assertIsNotNone(new_state["composer"].keras_model)
    
    def test_update_repository(self):
        # Test that update_repository adds a state if performance improves.
        state = self.agent.get_initial_state()
        state["performance"] = 0.5  # Simulate an improved performance.
        initial_repo_len = len(self.agent.repository)
        self.agent.update_repository(state)
        self.assertEqual(len(self.agent.repository), initial_repo_len + 1)
        self.assertAlmostEqual(self.agent.best_mse, 0.5)
    
    def test_apply_add_from_repository_action(self):
        # Ensure that when there is a repository entry, the add_from_repository action works.
        state = self.agent.get_initial_state()
        # First, update repository with an improved state.
        state["performance"] = 0.4
        self.agent.update_repository(state)
        actions_before = state["graph_actions"].copy()
        available_actions = self.agent.get_available_actions(state)
        self.assertIn("add_from_repository", available_actions)
        
        new_state = self.agent.apply_action(state, "add_from_repository")
        # Verify that the action history includes the "add_from_repository" action.
        self.assertIn("add_from_repository", new_state["graph_actions"])
        # Build the updated graph to ensure the model can be rebuilt.
        new_state["composer"].build()
        self.assertIsNotNone(new_state["composer"].keras_model)
    
    def test_apply_delete_repository_entry_action(self):
        # Add a dummy repository entry to enable deletion.
        self.agent.repository.append({"dummy": True})
        state = self.agent.get_initial_state()
        initial_repo_len = len(self.agent.repository)
        new_state = self.agent.apply_action(state, "delete_repository_entry")
        self.assertEqual(len(self.agent.repository), initial_repo_len - 1)

if __name__ == '__main__':
    unittest.main()
