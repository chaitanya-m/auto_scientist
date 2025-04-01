import unittest
import numpy as np
from function_graph.data_gen.curriculum import Curriculum

class TestCurriculum(unittest.TestCase):
    def test_reference_consistency(self):
        """Verify different seeds produce different models"""
        curriculum = Curriculum(seeds_per_phase=3)
        ref0 = curriculum.get_reference(0, 0)
        ref1 = curriculum.get_reference(0, 1)
        self.assertFalse(np.allclose(ref0['weights'][0], ref1['weights'][0]))

    def test_reference_performance(self):
        """Verify MSE meets quality threshold"""
        curriculum = Curriculum(seeds_per_phase=10)
        for seed in range(10):
            ref = curriculum.get_reference(0, seed) # for phase 0. Phase 1 can be tested on a bigger machine.
            self.assertLessEqual(ref['mse'], 0.1, 
                f"Seed {seed} MSE {ref['mse']:.4f} exceeds 0.1 threshold")

    def test_seed_variation(self):
        """Validate MSE variation between seeds"""
        curriculum = Curriculum(seeds_per_phase=10)
        mse_values = [curriculum.get_reference(0, seed)['mse'] for seed in range(10)]
        
        max_diff = np.max(mse_values) - np.min(mse_values)
        self.assertLessEqual(max_diff, 0.1, 
            f"MSE variation between seeds is {max_diff:.4f} (>0.1)")
        
        # Additional statistical checks
        self.assertLess(np.mean(mse_values), 0.1, "Mean MSE too high")
        self.assertLess(np.std(mse_values), 0.1, "MSE variance too high")
