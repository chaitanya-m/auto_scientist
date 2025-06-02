import unittest
import pandas as pd

from curriculum_generator.problems import AutoEncoderProblem
from self_directed_curriculum_learning.env.fg_env import FunctionGraphEnv
from run_experiment import run_simple_experiment

class QuickExperimentTest(unittest.TestCase):
    def test_env_step_and_observation(self):
        problem = AutoEncoderProblem(phase="basic", seed=0)
        env = FunctionGraphEnv(problem=problem, seed=0)
        obs, _ = env.reset()

        # Observation must be a 3-vector of floats
        self.assertEqual(obs.shape, (3,))
        self.assertTrue(all(isinstance(x, float) for x in obs))

        # pick any valid action and step once
        action = env.valid_actions()[0]
        obs2, reward, done, truncated, info = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertFalse(truncated)

        # After step, _get_obs() still returns 3-vector
        obs3 = env._get_obs()
        self.assertEqual(obs3.shape, (3,))

        # repository and composer must exist
        self.assertTrue(hasattr(env, "repository"))
        self.assertTrue(hasattr(env, "composer"))

    def test_run_simple_experiment_structure(self):
        problem = AutoEncoderProblem(phase="basic", seed=0)
        df, summary = run_simple_experiment(
            problem=problem,
            seed=0,
            mcts_budget=1,
            steps=1
        )

        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)

        expected_cols = {
            "seed", "step", "action", "did_reuse", "cumulative_reuse",
            "mse", "improvement", "nodes", "repo_size",
            "improvement_count", "deletion_count", "reward"
        }
        self.assertTrue(
            expected_cols.issubset(df.columns),
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

        # Sanity on the single row
        row = df.iloc[0]
        self.assertEqual(row.seed, 0)
        self.assertEqual(row.step, 0)
        self.assertIn(
            row.action,
            FunctionGraphEnv(problem=problem, seed=0).valid_actions()
        )
        self.assertIn(row.cumulative_reuse, (0, 1))
        self.assertGreaterEqual(row.nodes, 2)
        self.assertGreaterEqual(row.repo_size, 0)

        # Summary dict keys
        for key in ("steps_to_epsilon", "total_reuse",
                    "final_mse", "final_nodes", "final_repo_size"):
            self.assertIn(key, summary)

if __name__ == '__main__':
    unittest.main()
