# tests/test_mcts.py
import unittest
import numpy as np
import os
import uuid

from utils.graph_utils import compute_complexity
from utils.nn import create_minimal_graphmodel
from graph.node import SubGraphNode
from envs import FunctionGraphEnv
from agents.mcts import SimpleMCTSAgent

class TestGraphUtils(unittest.TestCase):
    def test_compute_complexity(self):
        """
        Test that compute_complexity returns the correct number of nodes in a GraphComposer.
        """
        composer, _ = create_minimal_graphmodel((3,), output_units=1, activation="relu")
        # By default: "input" and "output"
        self.assertEqual(compute_complexity(composer), 2)


class TestFunctionGraphEnv(unittest.TestCase):
    def setUp(self):
        self.env = FunctionGraphEnv(phase="basic", seed=0)

    def test_reset_observation(self):
        """
        After reset, the observation should be a length-3 numpy array
        matching [current_mse, #nodes, #actions].
        """
        obs, _ = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (3,))
        # initially current_mse is 1.0, nodes=2, actions=0
        np.testing.assert_allclose(obs, [1.0, 2.0, 0.0], rtol=1e-6)

    def test_valid_actions_initial(self):
        """
        Initially, repository is empty, so valid_actions() should be [0,1].
        """
        self.env.reset()
        self.assertListEqual(self.env.valid_actions(), [0, 1])

    def test_step_add_neuron_and_delete(self):
        """
        After add_neuron, composer grows; delete_repository_entry on non-empty repo
        should be valid only once repository has entries.
        """
        # Step 0: add_neuron
        self.env.reset()
        obs, reward, done, truncated, info = self.env.step(0)
        # still no repo entries (we train but may not improve), but step succeeds
        self.assertFalse(done)
        # delete is always valid even if repo empty
        self.assertIn(1, self.env.valid_actions())
        # reuse is invalid if no repo
        self.assertIn(2, self.env.valid_actions(),
              "After the first step, the improved candidate should be auto‐added to the repository, making action 2 valid.")


        # Manually inject a dummy repository entry to test reuse
        # Create a trivial SubGraphNode
        composer, model = create_minimal_graphmodel((self.env.input_dim,), output_units=self.env.latent_dim, activation="relu")
        tmp = f"/tmp/tmp_{uuid.uuid4().hex}.keras"
        composer.save_subgraph(tmp)
        sub = SubGraphNode.load(tmp, name="test")
        os.remove(tmp)
        self.env.repository.append({"subgraph_node": sub, "utility": -0.1})

        # Now reuse (action 2) should be valid
        self.assertIn(2, self.env.valid_actions())

    def test_step_invalid_action_raises(self):
        """
        Attempting to step with an invalid action index should raise an AssertionError.
        """
        self.env.reset()
        with self.assertRaises(AssertionError):
            self.env.step(2)  # reuse invalid initially


class TestSimpleMCTSAgent(unittest.TestCase):
    def setUp(self):
        # Create a fresh environment
        self.env = FunctionGraphEnv(phase="basic", seed=0)
        # Build a dummy policy network: just map observations to uniform probabilities
        import tensorflow as tf
        from keras import layers, models
        model = models.Sequential([
            layers.Input(shape=self.env.observation_space.shape),
            layers.Dense(self.env.action_space.n, activation="softmax")
        ])
        self.agent = SimpleMCTSAgent(env=self.env, policy_model=model, search_budget=5, c=1.0)

    def test_mcts_search_returns_valid_action(self):
        """
        Run a short MCTS search and verify the returned action is within valid_actions().
        """
        action = self.agent.mcts_search()
        valids = self.env.valid_actions()
        self.assertIsInstance(action, int)
        self.assertIn(action, valids,
                      f"MCTS returned invalid action {action}, valid are {valids}")


if __name__ == '__main__':
    unittest.main()
