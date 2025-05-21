# data_gen/problems.py

from abc import ABC, abstractmethod
import numpy as np
from sklearn.datasets import make_blobs

import tensorflow as tf
from typing import Dict, Callable


DEFAULT_PHASES = {
    'basic': {
        'input_dim': 3,
        'encoder': [3, 4, 2],  # 2 latent neurons (1 cluster × 2)
        'decoder': [2, 4, 3],
        'noise_level': 0.1,
        'clusters': 1,
        'cluster_std': 0.3  # single value is acceptable for one cluster
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


class Problem(ABC):
    """
    Abstract interface for a ground‐truth task.

    Implementations must provide:
      - sample_batch(): draw a batch of data (train & val)
      - reference_output(): return the true outputs for a batch
      - reference_mse: the “target” error scale (for normalization)
      - reference_complexity(): a scalar measuring ground‐truth complexity
      - input_dim / output_dim: dimensionalities of X and Y
    """
    @abstractmethod
    def sample_batch(self, batch_size: int):
        """
        Return ((X_train, Y_train), (X_val, Y_val)), where
        X are inputs and Y are targets drawn from the distribution.
        """
        pass

    @abstractmethod
    def reference_output(self, X_val):
        """
        Given validation inputs X_val, return the corresponding Y_val.
        """
        pass

    @property
    @abstractmethod
    def reference_mse(self) -> float:
        """
        A scalar error (e.g. pretrained model MSE) for normalizing rewards.
        """
        pass

    @abstractmethod
    def reference_complexity(self) -> float:
        """
        A scalar measuring the complexity of the ground‐truth mapping.
        """
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Dimensionality of the input X.
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Dimensionality of the prediction Y.
        """
        pass


class AutoencoderProblem(Problem):
    """
    Wraps the reference autoencoder as a Problem.
    This class now defines its own data generator.
    """
    def __init__(self, phase: str = "basic", problem_seed: int = 0, batch_size: int = 100):
        config = DEFAULT_PHASES[phase]
        # Validate architecture directly.
        self._validate_architecture(config)
        self.config = config
        

        self.problem_seed = problem_seed
        self._batch_counter = 0

        # Use the moved get_reference functionality (now in _get_reference)
        ref = self._get_reference(0, self.problem_seed, self._generate_data)
        self._encoder = ref["encoder"]
        self._mse = ref["mse"]
        self._complexity = len(ref["config"]["encoder"])
        self.batch_size = batch_size

        # Record dimensions.
        ref_cfg = ref["config"]
        self._input_dim = ref_cfg["input_dim"]
        self._output_dim = ref_cfg["encoder"][-1]  # latent dimension


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


    def _get_reference(self, phase: int, seed: int, data_generator) -> dict:
        """
        Retrieves the precomputed reference autoencoder configuration.
        Implements caching behavior previously provided by Curriculum.get_reference.
        """
        # Initialize the cache once.
        if not hasattr(self, '_reference_cache'):
            self._reference_cache = {}
        key = (phase, seed)
        if key not in self._reference_cache:
            self._reference_cache[key] = self._train_reference_autoencoder(self.config, seed, data_generator)
        return self._reference_cache[key]

    def _validate_architecture(self, config: dict):
        """
        Ensure latent dimension matches cluster requirements (2 per cluster)
        """
        latent_dim = config['encoder'][-1]
        required_dim = config['clusters'] * 2
        if latent_dim != required_dim:
            raise ValueError(f"Latent dim {latent_dim} should be {required_dim} for {config['clusters']} clusters")

    def _generate_data(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        Generates synthetic data for the autoencoder problem.
        """
        config = self.config
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=config['input_dim'],
            centers=config['clusters'],
            cluster_std=config['cluster_std'],
            random_state=seed
        )
        # Normalize data to [0, 1]
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X = (X - X_min) / (X_range + 1e-8)
        # Add noise if specified.
        if config['noise_level'] > 0:
            X += np.random.normal(scale=config['noise_level'], size=X.shape)
        return X

    def sample_batch(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        # Advance seed each call to get fresh data.
        sample_seed = self.problem_seed + self._batch_counter # Each problem may be sampled multiple times.
        self._batch_counter += 1

        # Use the internal generator.
        X_full = self._generate_data(bs, sample_seed)
        split = int(0.8 * len(X_full))
        Xtr, Xte = X_full[:split], X_full[split:]
        # Autoencoder tasks use X as both input and (placeholder) output.
        return (Xtr, None), (Xte, None)

    def reference_output(self, X_val):
        # Use the pretrained encoder to obtain ground-truth latent representations.
        return self._encoder.predict(X_val)

    @property
    def reference_mse(self) -> float:
        return self._mse

    def reference_complexity(self) -> float:
        return self._complexity

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim



class RegressionProblem(Problem):
    """
    Example: user plugs in any joint sampler and specifies
    which variables are inputs vs. targets.
    """
    def __init__(self, sampler_fn, input_vars, target_vars, batch_size: int = 100):
        """
        sampler_fn(batch_size) -> dict mapping variable names to np.arrays
        input_vars: list of keys for inputs X
        target_vars: list of keys for targets Y.
        """
        self.sampler_fn = sampler_fn
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.batch_size = batch_size
        self._last_val_Y = None

    def sample_batch(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        data = self.sampler_fn(bs)
        X = np.stack([data[k] for k in self.input_vars], axis=1)
        Y = np.stack([data[k] for k in self.target_vars], axis=1)
        split = int(0.8 * bs)
        Xtr, Ytr = X[:split], Y[:split]
        Xte, Yte = X[split:], Y[split:]
        self._last_val_Y = Yte
        return (Xtr, Ytr), (Xte, Yte)

    def reference_output(self, X_val):
        return self._last_val_Y

    @property
    def reference_mse(self) -> float:
        return float(np.var(self._last_val_Y))

    def reference_complexity(self) -> float:
        return float(len(self.input_vars) + len(self.target_vars))

    @property
    def input_dim(self) -> int:
        return len(self.input_vars)

    @property
    def output_dim(self) -> int:
        return len(self.target_vars)
