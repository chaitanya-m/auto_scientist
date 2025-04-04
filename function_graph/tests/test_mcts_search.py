import unittest
import tensorflow as tf
import numpy as np
from utils.nn import create_minimal_graphmodel
from agents.mcts import SimpleMCTSAgent
from graph.composer import GraphComposer

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
        # Build the model and check that its output dimension equals latent_dim.
        state["composer"].build()
        output_dim = state["composer"].keras_model.output.shape[-1]
        self.assertEqual(output_dim, self.agent.latent_dim)
    
    def test_evaluate_state_output_shape(self):
        """
        Test that after evaluation, the candidate encoder produces outputs with shape
        [batch_size, latent_dim].
        """
        state = self.agent.get_initial_state()
        mse = self.agent.evaluate_state(state)
        model = state["composer"].build()
        X, _ = self.agent.get_training_data()
        output = model.predict(X)
        self.assertEqual(output.shape[-1], self.agent.latent_dim,
                         "Candidate encoder output dimension should equal latent_dim.")
        # Also check that the MSE is a non-negative number.
        self.assertGreaterEqual(mse, 0.0, "MSE should be non-negative.")
    
    def test_search_loop_returns_best_state(self):
        # Run the search with a small search budget.
        best_state = self.agent.mcts_search(search_budget=10)
        # The best state should include a performance value.
        self.assertIn("performance", best_state)
        # Since our initial dummy performance is 1.0, we expect an improved performance (lower MSE).
        self.assertLess(best_state["performance"], 1.0, "The best state's performance should be improved (lower than 1.0)")

    def test_tree_structure_updated(self):
        _ = self.agent.mcts_search(search_budget=5)
        # Check that the agent's tree root exists.
        self.assertIsNotNone(self.agent.root, "The agent should maintain a tree structure (root should not be None)")
        
    def test_policy_network_stub(self):
        # Test the policy network stub returns valid probabilities for a given state and action list.
        state = self.agent.get_initial_state()
        actions = self.agent.get_available_actions(state)
        probabilities = self.agent.policy_network(state, actions)
        # Ensure probabilities sum to 1 and there is one probability per action.
        self.assertAlmostEqual(sum(probabilities), 1.0, msg="Probabilities should sum to 1")
        self.assertEqual(len(probabilities), len(actions), "There should be a probability for each action")

if __name__ == '__main__':
    unittest.main()
