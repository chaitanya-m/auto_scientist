import unittest
from agents.mcts import SimpleMCTSAgent

class TestMCTSSearch(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleMCTSAgent()
    
    def test_search_loop_returns_best_state(self):
        # Run the search with a small search budget.
        best_state = self.agent.mcts_search(search_budget=10)
        # The best state should include a performance value.
        self.assertIn("performance", best_state)
        # Since our initial dummy performance is 1.0, we expect an improved performance (lower MSE).
        self.assertLess(best_state["performance"], 1.0, "The best state's performance should be improved (lower than 1.0)")

    def test_tree_structure_updated(self):
        # Run the search and verify that the agent has built a non-empty search tree.
        _ = self.agent.mcts_search(search_budget=5)
        # Assume that the agent stores the root node in self.agent.tree.
        self.assertIsNotNone(self.agent.tree, "The agent should maintain a tree structure after search")
        self.assertGreater(len(self.agent.tree), 0, "The tree structure should not be empty after search")
        
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
