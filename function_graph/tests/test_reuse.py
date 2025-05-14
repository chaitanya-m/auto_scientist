import unittest
import uuid
from envs import FunctionGraphEnv

class TestCycle(unittest.TestCase):
    def test_repeated_reuse_never_creates_cycles(self):
        """
        After an initial improvement, repeatedly apply the 'reuse' action
        and assert that the composer remains cycle-free each time.
        """
        env = FunctionGraphEnv(phase="basic", seed=0)
        
        # 1) Force one improvement so the repository is non-empty
        #    by adding a neuron (action 0) and retraining.
        obs, reward, done, truncated, info = env.step(0)
        self.assertTrue(env.repository,
                        "Repository should have at least one entry after the first improvement.")

        # 2) Repeatedly invoke reuse (action 2) and verify no cycles
        for i in range(5):
            with self.subTest(iteration=i):
                # Step with reuse
                obs, reward, done, truncated, info = env.step(2)

                # Try a topological sort to detect cycles
                try:
                    env.composer._topological_sort()
                except ValueError as exc:
                    self.fail(f"Cycle detected on reuse iteration {i}: {exc}")

                # Optionally, ensure that each new subgraph node has a unique name
                names = [n for n in env.composer.nodes.keys() if n.startswith("sub_")]
                self.assertEqual(len(names), len(set(names)),
                                 "Subgraph node names should remain unique after reuse.")

if __name__ == "__main__":
    unittest.main()
