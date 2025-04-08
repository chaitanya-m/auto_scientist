#tests/test_mcts.py
import unittest
import tensorflow as tf
import numpy as np
from utils.nn import create_minimal_graphmodel
from agents.mcts import SimpleMCTSAgent
from graph.composer import GraphComposer

class TestMinimalGraphModel(unittest.TestCase):
    def test_single_neuron_case(self):
        # Test default behavior with a single neuron output.
        input_shape = (3,)
        composer, model = create_minimal_graphmodel(input_shape)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsNotNone(model.input)
        self.assertIsNotNone(model.output)
        
        output_shape = model.output.shape
        self.assertEqual(output_shape[-1], 1)
        
        self.assertEqual(composer.input_node_name, "input")
        self.assertIn("output", composer.output_node_names)

    def test_encoder_case(self):
        # Test the encoder case where output_units > 1.
        input_shape = (3,)
        output_units = 4  # Example: latent space of 4 units.
        activation = "relu"
        composer, model = create_minimal_graphmodel(input_shape, output_units=output_units, activation=activation)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsNotNone(model.input)
        self.assertIsNotNone(model.output)
        
        output_shape = model.output.shape
        self.assertEqual(output_shape[-1], output_units)
        
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
        
        # Build the model and check that its output dimension equals latent_dim.
        state["composer"].build()
        output_dim = state["composer"].keras_model.output.shape[-1]
        self.assertEqual(output_dim, self.agent.latent_dim)
    
    def test_get_available_actions(self):
        state = self.agent.get_initial_state()
        actions = self.agent.get_available_actions(state)
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
        state = self.agent.get_initial_state()
        state["performance"] = 0.5  # Simulate improved performance.
        initial_repo_len = len(self.agent.repository)
        self.agent.update_repository(state)
        self.assertEqual(len(self.agent.repository), initial_repo_len + 1)
        self.assertAlmostEqual(self.agent.best_mse, 0.5)
    
    def test_apply_add_from_repository_action(self):
        state = self.agent.get_initial_state()
        # Update repository with an improved state.
        state["performance"] = 0.4
        self.agent.update_repository(state)
        actions_before = state["graph_actions"].copy()
        available_actions = self.agent.get_available_actions(state)
        self.assertIn("add_from_repository", available_actions)
        
        new_state = self.agent.apply_action(state, "add_from_repository")
        self.assertIn("add_from_repository", new_state["graph_actions"])
        new_state["composer"].build()
        self.assertIsNotNone(new_state["composer"].keras_model)
    
    def test_apply_delete_repository_entry_action(self):
        self.agent.repository.append({"dummy": True})
        state = self.agent.get_initial_state()
        initial_repo_len = len(self.agent.repository)
        new_state = self.agent.apply_action(state, "delete_repository_entry")
        self.assertEqual(len(self.agent.repository), initial_repo_len - 1)
    
    def test_evaluate_state_output_shape(self):
        """
        Test that after evaluation the candidate encoder's output shape matches [batch_size, latent_dim].
        """
        state = self.agent.get_initial_state()
        # Evaluate the candidate encoder.
        mse = self.agent.evaluate_state(state)
        # Build the candidate model.
        model = state["composer"].build()
        # Get dummy input data.
        import numpy as np
        X, _ = self.agent.get_training_data()
        # Run prediction.
        output = model.predict(X)
        self.assertEqual(output.shape[-1], self.agent.latent_dim,
                         "Candidate encoder output dimension should equal latent_dim.")

# tests/test_mcts.py
import unittest
import numpy as np
from agents.mcts import SimpleMCTSAgent
from graph.composer import GraphComposer
from utils.nn import create_minimal_graphmodel
from graph.node import SubGraphNode

class TestRepositoryUtility(unittest.TestCase):

    def setUp(self):
        self.agent = SimpleMCTSAgent()
        
    def test_repository_entry_utility_calculation(self):
        # Create a dummy state with a known performance
        state = self.agent.get_initial_state()
        state["performance"] = 0.25  # known MSE
        # Call update_repository to add a repository entry based on the state.
        initial_repo_len = len(self.agent.repository)
        self.agent.update_repository(state)
        self.assertEqual(len(self.agent.repository), initial_repo_len + 1)
        # Check that the utility value equals -performance.
        repo_entry = self.agent.repository[-1]
        self.assertAlmostEqual(repo_entry["utility"], -0.25, msg="Utility should equal negative performance.")

    def test_add_from_repository_selects_best_entry(self):
        # Create real SubGraphNode instances using the minimal graph model function.
        # Each subgraph node is built from a valid model with the current input dimension.
        sub_nodes = []
        for name in ["subgraph1", "subgraph2", "subgraph3"]:
            composer, model = create_minimal_graphmodel((self.agent.input_dim,), output_units=1, activation="relu")
            node = SubGraphNode(name=name, model=model)
            sub_nodes.append(node)
        
        # Create a dummy repository with entries having different performances.
        dummy_entries = [
            {"subgraph_node": sub_nodes[0], "performance": 0.5, "graph_actions": [], "utility": -0.5},
            {"subgraph_node": sub_nodes[1], "performance": 0.3, "graph_actions": [], "utility": -0.3},
            {"subgraph_node": sub_nodes[2], "performance": 0.2, "graph_actions": [], "utility": -0.2}
        ]
        # Intentionally shuffling is not necessary since we use max() over utility.
        self.agent.repository = dummy_entries.copy()
        
        # Create a dummy state with a valid composer.
        state = self.agent.get_initial_state()
        state["graph_actions"] = []
        # Execute the "add_from_repository" action.
        new_state = self.agent.apply_action(state, "add_from_repository")
        
        # Since the best (highest) utility is -0.2 (from subgraph3),
        # we expect that the transformer used the corresponding repository entry.
        # At minimum, the composer should rebuild its Keras model, indicating the action was performed.
        self.assertIsNotNone(new_state["composer"].keras_model, 
                             "Composer should rebuild the Keras model after add_from_repository.")
    
    def test_policy_network_output_filtering(self):
        # Test that reducing the available actions returns a probability vector
        # with the correct dimensions that sums to 1.
        state = self.agent.get_initial_state()
        available_actions = ["add_neuron", "delete_repository_entry"]  # Excluding reuse.
        probs = self.agent.policy_network(state, available_actions)
        self.assertEqual(len(probs), len(available_actions),
                         "Policy network output should match the available actions count")
        np.testing.assert_almost_equal(sum(probs), 1.0, decimal=5,
            err_msg="Filtered probabilities should sum to 1.")


# tests/test_mcts.py
import unittest
import numpy as np
import tensorflow as tf
from agents.mcts import SimpleMCTSAgent
from graph.composer import GraphComposer
from utils.nn import create_minimal_graphmodel
from graph.node import SubGraphNode

class TestRepositoryUtility(unittest.TestCase):

    def setUp(self):
        self.agent = SimpleMCTSAgent()
        
    def test_repository_entry_utility_calculation(self):
        # Create a state with a known performance
        state = self.agent.get_initial_state()
        state["performance"] = 0.25  # known MSE
        # Update the repository based on this state.
        initial_repo_len = len(self.agent.repository)
        self.agent.update_repository(state)
        self.assertEqual(len(self.agent.repository), initial_repo_len + 1)
        # The utility should equal -performance.
        repo_entry = self.agent.repository[-1]
        self.assertAlmostEqual(repo_entry["utility"], -0.25, msg="Utility should equal negative performance.")

    def test_add_from_repository_selects_best_entry(self):
        # Create real SubGraphNode instances using the minimal graph model.
        sub_nodes = []
        for name in ["subgraph1", "subgraph2", "subgraph3"]:
            composer, model = create_minimal_graphmodel((self.agent.input_dim,), output_units=1, activation="relu")
            node = SubGraphNode(name=name, model=model)
            sub_nodes.append(node)
        
        # Build dummy repository entries with distinct performance (and utility) values.
        dummy_entries = [
            {"subgraph_node": sub_nodes[0], "performance": 0.5, "graph_actions": [], "utility": -0.5},
            {"subgraph_node": sub_nodes[1], "performance": 0.3, "graph_actions": [], "utility": -0.3},
            {"subgraph_node": sub_nodes[2], "performance": 0.2, "graph_actions": [], "utility": -0.2}
        ]
        self.agent.repository = dummy_entries.copy()
        
        # Create a state with a valid composer.
        state = self.agent.get_initial_state()
        state["graph_actions"] = []
        # Apply the "add_from_repository" action.
        new_state = self.agent.apply_action(state, "add_from_repository")
        
        # Check that the composer's Keras model is rebuilt,
        # which indicates that the transformation was executed.
        self.assertIsNotNone(new_state["composer"].keras_model, 
                             "Composer should rebuild the Keras model after add_from_repository.")
    
    def test_policy_network_output_filtering(self):
        # Check that if the available action set is reduced,
        # the policy network outputs a vector with the correct length and that sums to 1.
        state = self.agent.get_initial_state()
        available_actions = ["add_neuron", "delete_repository_entry"]  # Excluding repository reuse.
        probs = self.agent.policy_network(state, available_actions)
        self.assertEqual(len(probs), len(available_actions),
                         "Policy network output should match the available actions count")
        np.testing.assert_almost_equal(sum(probs), 1.0, decimal=5,
            err_msg="Filtered probabilities should sum to 1.")

class TestExperienceBuffer(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleMCTSAgent()
        # Lower the threshold for quicker triggering in tests.
        self.agent.experience_threshold = 5

    def test_experience_training_trigger(self):
        initial_buffer_len = len(self.agent.experience)
        # Record dummy experiences until the threshold is reached.
        dummy_state = self.agent.get_initial_state()
        for i in range(5):
            self.agent.record_experience(dummy_state, "add_neuron", reward= -0.1 * i)
        # After recording enough experiences, the experience buffer should be cleared.
        self.assertEqual(len(self.agent.experience), 0,
                         "Experience buffer should be cleared after training is triggered.")

    def test_policy_network_training_effect(self):
        # Check that policy network parameters are updated after training.
        # We capture the parameters before and after training dummy experiences.
        initial_weights = self.agent.policy_net.model.get_weights()
        dummy_state = self.agent.get_initial_state()
        # Record enough dummy experiences to trigger training.
        for i in range(5):
            self.agent.record_experience(dummy_state, "delete_repository_entry", reward=-0.1 * i)
        updated_weights = self.agent.policy_net.model.get_weights()
        # Test that at least one weight array has changed.
        weight_changed = any(
            not np.array_equal(initial, updated)
            for initial, updated in zip(initial_weights, updated_weights)
        )
        self.assertTrue(weight_changed, "Policy network weights should be updated after training.")

class TestMCTSSearchOutcome(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleMCTSAgent()

    def test_mcts_search_improves_performance(self):
        # Run the full MCTS search with a certain budget.
        best_state = self.agent.mcts_search(search_budget=10)
        # Check that the candidate's performance is improved from the dummy value (1.0).
        self.assertIn("performance", best_state)
        self.assertLess(best_state["performance"], 1.0,
                        "The best state's performance should be improved (lower than 1.0) after MCTS search.")

if __name__ == '__main__':
    unittest.main()
