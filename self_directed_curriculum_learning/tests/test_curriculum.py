#tests/test_curriculum.py
import unittest
import numpy as np
import tensorflow as tf
from curriculum_generator.curriculum import Curriculum

class TestCurriculum(unittest.TestCase):
    def test_reference_consistency(self):
        """Verify different seeds produce different models."""
        curriculum = Curriculum(seeds_per_phase=3)
        ref0 = curriculum.get_reference(0, 0)
        ref1 = curriculum.get_reference(0, 1)
        # Check that the first layer's weights are not nearly identical.
        self.assertFalse(np.allclose(ref0['autoencoder'].get_weights()[0],
                                      ref1['autoencoder'].get_weights()[0]),
                         "Different seeds should yield different weight initializations.")

    def test_reference_performance(self):
        """Verify MSE meets quality threshold."""
        curriculum = Curriculum(seeds_per_phase=10)
        for seed in range(10):
            ref = curriculum.get_reference(0, seed)  # Testing phase 0.
            self.assertLessEqual(ref['mse'], 0.1, 
                f"Seed {seed} MSE {ref['mse']:.4f} exceeds 0.1 threshold")

    def test_seed_variation(self):
        """Validate MSE variation between seeds."""
        curriculum = Curriculum(seeds_per_phase=10)
        mse_values = [curriculum.get_reference(0, seed)['mse'] for seed in range(10)]
        max_diff = np.max(mse_values) - np.min(mse_values)
        self.assertLessEqual(max_diff, 0.1, 
            f"MSE variation between seeds is {max_diff:.4f} (>0.1)")
        self.assertLess(np.mean(mse_values), 0.1, "Mean MSE too high")
        self.assertLess(np.std(mse_values), 0.1, "MSE variance too high")
    
    def test_reference_keys(self):
        """Verify that reference autoencoder contains expected keys and models."""
        curriculum = Curriculum(seeds_per_phase=3)
        ref = curriculum.get_reference(0, 0)
        expected_keys = ['mse', 'autoencoder', 'encoder', 'decoder', 'config', 'seed']
        for key in expected_keys:
            self.assertIn(key, ref, f"Key '{key}' not found in reference autoencoder.")
        
        # Check that autoencoder, encoder, and decoder are Keras models.
        self.assertTrue(isinstance(ref['autoencoder'], tf.keras.Model),
                        "autoencoder is not a valid Keras Model.")
        self.assertTrue(isinstance(ref['encoder'], tf.keras.Model),
                        "encoder is not a valid Keras Model.")
        self.assertTrue(isinstance(ref['decoder'], tf.keras.Model),
                        "decoder is not a valid Keras Model.")
        
        # Check that the seed matches.
        self.assertEqual(ref['seed'], 0, "Returned seed does not match expected value.")

    def test_data_generator_output(self):
        """Verify that the data generator returns data of expected shape and variability."""
        curriculum = Curriculum(seeds_per_phase=3)
        # Request 50 samples.
        X1 = curriculum.data_generator(50)
        self.assertEqual(X1.shape, (50, curriculum.phase_config['input_dim']),
                         "Data generator did not return expected shape.")
        # Request another 50 samples without a fixed seed; ensure the output is different.
        X2 = curriculum.data_generator(50)
        # It's possible (though unlikely) that the outputs are identical; we'll check that they are not almost equal.
        self.assertFalse(np.allclose(X1, X2),
                         "Data generator should produce variable outputs when no seed is provided.")

if __name__ == '__main__':
    unittest.main()
