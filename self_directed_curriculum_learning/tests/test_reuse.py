import unittest
import uuid
from env.fg_env import FunctionGraphEnv
from curriculum_generator.curriculum_interface import CurriculumInterface

# Mock curriculum for testing
class MockCurriculum(CurriculumInterface):
    def __init__(self, seed=0):
        self.difficulty = 1.0
        self._seed = seed
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Return a mock problem with required attributes
        class MockProblem:
            def __init__(self):
                self.reference_mse = 1.0
                self.input_dim = 10
                self.output_dim = 5
                
            def reference_complexity(self):
                return 10.0
                
            def sample_batch(self, batch_size):
                import numpy as np
                X = np.random.randn(batch_size, self.input_dim)
                Y = np.random.randn(batch_size, self.output_dim)
                return (X, Y), (X, Y)
                
            def reference_output(self, X):
                import numpy as np
                return np.random.randn(X.shape[0], self.output_dim)
        
        return MockProblem()

class TestCycle(unittest.TestCase):
    def test_repeated_reuse_never_creates_cycles(self):
        """
        After an initial improvement, repeatedly apply the 'reuse' action
        and assert that the composer remains cycle-free each time.
        """
        # Create mock curriculum
        curriculum = MockCurriculum(seed=0)
        env = FunctionGraphEnv(curriculum=curriculum, train_epochs=1, batch_size=10, seed=0)
        
        # Reset environment to initialize
        env.reset()
        
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
