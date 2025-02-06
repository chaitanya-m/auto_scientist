import unittest
import numpy as np
from utils.rl_dev import RLEnvironment, DummyAgent, run_episode

class TestRLEnvironmentDatasetReuse(unittest.TestCase):
    def test_dataset_consistency_and_reuse(self):
        """
        Test that in a fixed-distribution episode, the environment provides the same dataset at each step.
        """
        env = RLEnvironment(total_steps=1000, num_instances_per_step=100, seed=0)
        agent = DummyAgent()
        actions, rewards = run_episode(env, agent)

        # Retrieve state at final step; the dataset should be identical to the initial one.
        state_initial = env._get_state()  # After episode run, state remains at final step.
        
        # Reset environment to get the original dataset
        env.reset()
        state_reset = env._get_state()

        # Check that the dataset remains constant.
        # Here we compare the DataFrame shapes and first few rows.
        dataset_initial = state_reset["dataset"]
        dataset_final = state_initial["dataset"]
        
        self.assertEqual(dataset_initial.shape, dataset_final.shape,
                         "Dataset shape should remain constant across steps")
        # Compare a few values to ensure consistency.
        for col in dataset_initial.columns:
            np.testing.assert_array_equal(dataset_initial[col].values, dataset_final[col].values,
                                          err_msg="Dataset values should be identical across steps")

if __name__ == '__main__':
    unittest.main()
