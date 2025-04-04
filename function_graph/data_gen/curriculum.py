#data_gen/curriculum.py
import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf
from typing import Dict, List, Tuple

class Curriculum:
    DEFAULT_PHASES = {
        'basic': {
            'input_dim': 3,
            'encoder': [3, 4, 2],  # 2 latent neurons (1 cluster × 2)
            'decoder': [2, 4, 3],
            'noise_level': 0.1,
            'clusters': 1,
            'cluster_std': [0.3]
        },
        'intermediate': {
            'input_dim': 3,
            'encoder': [3, 6, 4],  # 4 latent neurons (2 clusters × 2)
            'decoder': [4, 6, 3],
            'noise_level': 0.2,
            'clusters': 2,
            'cluster_std': [0.3, 0.5]
        }
    }

    def __init__(self, phase_type='basic', seeds_per_phase=10):
        """
        Manages a curriculum of autoencoder tasks with validated architecture
        
        Args:
            phase_type: Which phase configuration to use (basic/intermediate)
            seeds_per_phase: Number of random seeds for reproducibility
        """
        self.phase_config = self.DEFAULT_PHASES[phase_type]
        self.seeds_per_phase = seeds_per_phase
        self.reference_cache = {}
        self._validate_architecture()
        self._precompute_references()

    def _validate_architecture(self):
        """Ensure latent dimension matches cluster requirements (2 per cluster)"""
        latent_dim = self.phase_config['encoder'][-1]
        required_dim = self.phase_config['clusters'] * 2
        if latent_dim != required_dim:
            raise ValueError(f"Latent dim {latent_dim} should be {required_dim} for {self.phase_config['clusters']} clusters")

    def _precompute_references(self):
        """Precompute reference autoencoders for all seeds in current phase"""
        for seed in range(self.seeds_per_phase):
            key = (0, seed)  # Single phase index 0
            self.reference_cache[key] = self._train_reference_autoencoder(
                self.phase_config, 
                seed
            )


    def _train_reference_autoencoder(self, config: Dict, seed: int) -> Dict:
        """Train and store reference autoencoder and its components"""
        tf.keras.utils.set_random_seed(seed)
        

            # Generate synthetic data using Gaussian blobs (isotropic clusters)
            # Creates 3 separable clusters of normally distributed points:
            # - n_samples: 1000 total points (approx 333 per cluster)
            # - n_features: Dimension matches phase configuration input size
            # - centers: 3 cluster centers in feature space
            # - cluster_std: 1.0 standard deviation for each cluster
            # - random_state: Seed for reproducible data generation
        X, _ = make_blobs(
            n_samples=1000,
            n_features=config['input_dim'],
            centers=config['clusters'],
            cluster_std=config['cluster_std'],
            random_state=seed
        )
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X = (X - X_min) / (X_range + 1e-8)
        
        if config['noise_level'] > 0:
            X += np.random.normal(scale=config['noise_level'], size=X.shape)
        
        # Build autoencoder model.
        autoencoder = tf.keras.Sequential()
        
        # Encoder layers.
        for size in config['encoder']:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
        
        # Decoder layers.
        for size in config['decoder'][:-1]:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
        autoencoder.add(tf.keras.layers.Dense(config['decoder'][-1], activation='linear'))
        
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
        
        val_mse = min(history.history['val_loss'])
        
        # Extract encoder and decoder from the autoencoder.
        # Assuming a Sequential model, the encoder is the first len(config['encoder']) layers.
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.InputLayer(input_shape=(config['input_dim'],)))
        for i in range(len(config['encoder'])):
            encoder.add(autoencoder.layers[i])
        
        # The decoder is the remainder of the model.
        decoder = tf.keras.Sequential()
        # For the decoder, we need to infer the shape of the latent space from the encoder's output.
        latent_dim = config['encoder'][-1]
        decoder.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        for i in range(len(config['encoder']), len(autoencoder.layers)):
            decoder.add(autoencoder.layers[i])
        
        return {
            'mse': val_mse,
            'autoencoder': autoencoder,
            'encoder': encoder,
            'decoder': decoder,
            'config': config.copy(),
            'seed': seed
        }

    def get_reference(self, phase: int, seed: int) -> Dict:
        """Get precomputed reference for a phase/seed combination"""
        return self.reference_cache[(phase, seed)]

    @property
    def num_phases(self) -> int:
        return len(self.phases)
