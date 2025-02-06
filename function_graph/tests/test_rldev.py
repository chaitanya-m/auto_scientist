import unittest
import numpy as np
from utils.rl_dev import RLEnvironment, DummyAgent, run_episode

class TestAgentReuseTrend(unittest.TestCase):
    def test_reuse_rate_increases(self):
        """
        Hypothesis: Under a fixed data distribution, the agent's reuse rate trends upward.
        Test: Run 1000 steps and verify that the reuse rate in the last 10% of steps
        exceeds that in the first 10%.
        """
        env = RLEnvironment(total_steps=1000)
        agent = DummyAgent(total_steps=1000)
        actions = run_episode(env, agent)
        
        first_10_rate = actions[:100].count("reuse") / 100
        last_10_rate = actions[-100:].count("reuse") / 100
        
        print(f"First 10% reuse rate: {first_10_rate:.3f}, Last 10% reuse rate: {last_10_rate:.3f}")
        self.assertGreater(last_10_rate, first_10_rate)

if __name__ == '__main__':
    unittest.main()
