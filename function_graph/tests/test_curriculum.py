import unittest
import numpy as np
from function_graph.data_gen.curriculum import Curriculum

class TestCurriculum(unittest.TestCase):
    SEEDS_PER_PHASE = 2
    SAMPLE_PHASES = [
        {
            'input_dim': 2,
            'encoder': [2, 1],
            'decoder': [2, 2],
            'noise_level': 0.0
        }
    ]

    def test_curriculum_initialization(self):
        curriculum = Curriculum(phases=self.SAMPLE_PHASES, seeds_per_phase=self.SEEDS_PER_PHASE)
        
        self.assertEqual(curriculum.num_phases, 1)
        self.assertEqual(len(curriculum.reference_cache), 1 * self.SEEDS_PER_PHASE)  # 1 phase x seeds

    def test_reference_consistency(self):
        curriculum = Curriculum(phases=self.SAMPLE_PHASES, seeds_per_phase=2)  # Only need 2 seeds for comparison
        
        # Test first phase references
        phase0_ref0 = curriculum.get_reference(0, 0)
        phase0_ref1 = curriculum.get_reference(0, 1)
        
        # Same phase different seeds should have different weights
        self.assertFalse(np.allclose(
            phase0_ref0['weights'][0], 
            phase0_ref1['weights'][0]
        ))
        
        # Config should match phase specification
        self.assertEqual(
            phase0_ref0['config']['encoder'],
            self.SAMPLE_PHASES[0]['encoder']
        )


    def test_reference_performance(self):
        curriculum = Curriculum(phases=self.SAMPLE_PHASES, seeds_per_phase=self.SEEDS_PER_PHASE)
        
        # Test that references achieve reasonable MSE
        for phase in range(curriculum.num_phases):
            for seed in range(self.SEEDS_PER_PHASE):
                ref = curriculum.get_reference(phase, seed)
                mse = ref['mse']
            print(f"Phase {phase} Seed {seed}: MSE={mse:.6f}")  # Debug output
            self.assertLess(mse, 0.1, 
                    f"Phase {phase} seed {seed} MSE {mse:.4f} exceeds threshold")

    def test_seed_variation(self):
        curriculum = Curriculum(phases=self.SAMPLE_PHASES, seeds_per_phase=self.SEEDS_PER_PHASE)
        mse_values = []
        
        for phase in range(curriculum.num_phases):
            for seed in range(self.SEEDS_PER_PHASE):
                ref = curriculum.get_reference(phase, seed)
                mse_values.append(ref['mse'])
        
        # Validate statistical performance across seeds
        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        
        self.assertLess(mean_mse, 0.08, "Mean MSE exceeds threshold")
        self.assertLess(std_mse, 0.02, "MSE variance too high")
