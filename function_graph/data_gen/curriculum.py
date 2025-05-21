# data_gen/curriculum.py
import numpy as np
import tensorflow as tf
from typing import Dict, Callable

class Curriculum:
    def __init__(self, config: dict):
        """
        Manages a curriculum of autoencoder tasks.
        Args:
            config: Configuration dictionary.
        """
        self.config = config

    def _train_reference_autoencoder(self, config: Dict, seed: int, data_generator: Callable[[int, int], np.ndarray]) -> Dict:
        """Train and store reference autoencoder and its components using provided data."""
        tf.keras.utils.set_random_seed(seed)
        
        # Generate training data via the provided generator.
        X = data_generator(1000, seed=seed)
        
        # Build autoencoder.
        autoencoder = tf.keras.Sequential()
        for size in config['encoder']:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
        for size in config['decoder'][:-1]:
            autoencoder.add(tf.keras.layers.Dense(size, activation='relu'))
        autoencoder.add(tf.keras.layers.Dense(config['decoder'][-1], activation='linear'))
        
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        history = autoencoder.fit(
            X, X,
            epochs=10000,
            validation_split=0.2,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
        )
        val_mse = min(history.history['val_loss'])
        
        # Extract encoder.
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.InputLayer(input_shape=(config['input_dim'],)))
        for i in range(len(config['encoder'])):
            encoder.add(autoencoder.layers[i])
        
        # Extract decoder.
        decoder = tf.keras.Sequential()
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

    @property
    def num_phases(self) -> int:
        # With an externally provided config, there is a single phase.
        return 1
