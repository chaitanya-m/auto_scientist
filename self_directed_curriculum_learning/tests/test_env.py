import unittest
import numpy as np
from keras import models, layers

from env.fg_env import FunctionGraphEnv
from env.graph.node import SubGraphNode
from curriculum_generator.problems import Problem

class DummyProblem(Problem):
    """
    Minimal Problem stub for testing FunctionGraphEnv.
    """
    @property
    def reference_mse(self):
        return 0.5

    def reference_complexity(self):
        return 1.0

    @property
    def input_dim(self):
        return 3

    @property
    def output_dim(self):
        return 1

    def sample_batch(self, batch_size):
        # return trivial data
        X = np.zeros((batch_size, self.input_dim))
        Y = np.zeros((batch_size, self.output_dim))
        return (X, Y), (X, Y)

    def reference_output(self, X):
        return np.zeros((X.shape[0], self.output_dim))

    # Implement abstract methods required by Problem
    def get_phase_config(self):
        return {}

    def seeded_problem_variations(self, seed: int):
        return [self]

class TestFunctionGraphEnvClone(unittest.TestCase):
    def setUp(self):
        # Initialize env and add a dummy repository entry
        self.env = FunctionGraphEnv(problem=DummyProblem(), train_epochs=1, batch_size=10, seed=42)

        # Manually add a SubGraphNode entry to repository
        inp = layers.Input(shape=(1,))
        out = layers.Dense(1, activation='linear')(inp)
        bypass = models.Model(inputs=inp, outputs=out)
        sgn = SubGraphNode(name='test_sgn', model=bypass)
        entry = {'subgraph_node': sgn, 'utility': 0.1}
        self.env.repository.append(entry)

        # Record original state IDs
        self.orig_repo_id = id(self.env.repository)
        self.orig_entry_id = id(self.env.repository[0])

        # Set some scalar state
        self.env.best_mse = 0.2
        self.env.current_mse = 0.3
        self.env.graph_actions = ['a', 'b']
        self.env.deletion_count = 2
        self.env.improvement_count = 3

    def test_clone_repository_shallow_copy(self):
        clone = self.env.clone()
        # Repository list is a new object
        self.assertNotEqual(id(clone.repository), self.orig_repo_id)
        # Entry dict is the same object (shallow copy)
        self.assertEqual(id(clone.repository[0]), self.orig_entry_id)
        # Modifying original list does not affect clone
        self.env.repository.append({'subgraph_node': None, 'utility': 0})
        self.assertEqual(len(clone.repository), 1)

    def test_clone_composer_independence(self):
        clone = self.env.clone()
        # Composer objects are distinct
        self.assertIsNot(clone.composer, self.env.composer)
        # Keras models are distinct
        self.assertIsNot(clone.composer.keras_model, self.env.composer.keras_model)
        # But their structures are the same (layer count)
        orig_layers = [l.name for l in self.env.composer.keras_model.layers]
        clone_layers = [l.name for l in clone.composer.keras_model.layers]
        self.assertEqual(len(orig_layers), len(clone_layers))

        # Changing clone's composer should not affect original
        clone.composer.nodes.clear()
        self.assertNotEqual(len(self.env.composer.nodes), len(clone.composer.nodes))

    def test_scalar_state_copied(self):
        clone = self.env.clone()
        self.assertEqual(clone.best_mse, self.env.best_mse)
        self.assertEqual(clone.current_mse, self.env.current_mse)
        self.assertListEqual(clone.graph_actions, self.env.graph_actions)
        self.assertEqual(clone.deletion_count, self.env.deletion_count)
        self.assertEqual(clone.improvement_count, self.env.improvement_count)

if __name__ == '__main__':
    unittest.main()
