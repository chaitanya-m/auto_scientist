import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf
from typing import Dict, List, Tuple

class Curriculum:
    def __init__(self, phases: List[Dict], seeds_per_phase: int = 10):
        """
        Manages a curriculum of increasingly complex autoencoder tasks
        
        Args:
            phases: List of phase configurations, each specifying:
                - input_dim: Input dimension
                - encoder: List of encoder layer sizes
                - decoder: List of decoder layer sizes  
                - noise_level: Gaussian noise stddev
            seeds_per_phase: Number of random seeds per phase
        """
        self.phases = phases
        self.seeds_per_phase = seeds_per_phase
        self.reference_cache = {}  # Stores precomputed reference models
        
        # Precompute all reference autoencoders
        self._precompute_references()

    def _precompute_references(self):
        """Precompute reference autoencoders for all phases and seeds"""
        for phase_idx, phase_config in enumerate(self.phases):
            for seed in range(self.seeds_per_phase):
                key = (phase_idx, seed)
                self.reference_cache[key] = self._train_reference_autoencoder(
                    phase_config, 
                    seed
                )

    def _train_reference_autoencoder(self, config: Dict, seed: int) -> Dict:
        """Train and store reference autoencoder"""
        tf.keras.utils.set_random_seed(seed)
        
        # Generate and normalize synthetic data
        X, _ = make_blobs(
            n_samples=1000,
            n_features=config['input_dim'],
            centers=3,
            random_state=seed
        )
        # Min-max scaling to [0,1] range
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X = (X - X_min) / (X_range + 1e-8)  # Add small epsilon to prevent division by zero
        
        # Add noise if specified
        if config['noise_level'] > 0:
            X += np.random.normal(scale=config['noise_level'], size=X.shape)
        
        # Build autoencoder with bottleneck and final linear activation
        autoencoder = tf.keras.Sequential()
        
        # Encoder
        for size in config['encoder']:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
            
        # Decoder with linear output for normalized data
        for size in config['decoder'][:-1]:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
        autoencoder.add(tf.keras.layers.Dense(config['decoder'][-1], activation='linear'))
        
        # Train with better optimization
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        history = autoencoder.fit(
            X, X,
            epochs=10000,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=20, 
                    restore_best_weights=True
                )
            ]
        )
        
        return {
            'mse': min(history.history['val_loss']),
            'weights': autoencoder.get_weights(),
            'config': config.copy()
        }

    def get_reference(self, phase: int, seed: int) -> Dict:
        """Get precomputed reference for a phase/seed combination"""
        return self.reference_cache[(phase, seed)]

    @property
    def num_phases(self) -> int:
        return len(self.phases)
