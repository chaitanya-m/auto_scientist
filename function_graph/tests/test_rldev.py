import unittest
import numpy as np
import pandas as pd
from rl_dev import RLEnvironment, DummyAgent, run_episode

class TestRLEnvironmentAndNetwork(unittest.TestCase):
    def test_dataset_consistency(self):
        """
        Test that in a fixed-distribution episode, the environment provides the same dataset at each step.
        """
        env = RLEnvironment(total_steps=1000, num_instances_per_step=100, seed=0)
        # Reset the environment and capture the dataset.
        state_initial = env.reset()
        dataset_initial = state_initial["dataset"]

        # Run a few steps and check that the dataset remains unchanged.
        for _ in range(10):
            state, _, _ = env.step("new")
            dataset_current = state["dataset"]
            self.assertEqual(dataset_initial.shape, dataset_current.shape,
                             "Dataset shape should remain constant across steps")
            for col in dataset_initial.columns:
                np.testing.assert_array_equal(
                    dataset_initial[col].values, dataset_current[col].values,
                    err_msg="Dataset values should be identical across steps"
                )

    def test_minimal_network_structure(self):
        """
        Test that the minimal network created by the environment contains the expected nodes and connection.
        Expected nodes: 'input' and 'output'
        Expected connection: 'input' should be connected to 'output'
        """
        env = RLEnvironment(total_steps=1000, num_instances_per_step=100, seed=0)
        env.reset()
        state = env._get_state()
        network = state["network"]

        # Check that both 'input' and 'output' nodes exist.
        self.assertIn("input", network["nodes"], "Network should contain an 'input' node")
        self.assertIn("output", network["nodes"], "Network should contain an 'output' node")
        
        # Check that there is a connection from 'input' to 'output'
        connections = network.get("connections", {})
        self.assertIn("output", connections, "Output node should have incoming connections")
        connection_parents = [parent for parent, _ in connections["output"]]
        self.assertIn("input", connection_parents, "Output node should be connected from 'input'")

if __name__ == '__main__':
    unittest.main()
