# data_gen/problems.py

from abc import ABC, abstractmethod
import numpy as np
from data_gen.curriculum import Curriculum

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
    Wraps your Curriculum-based reference autoencoder as a Problem.
    """
    def __init__(self, phase: str = "basic", seed: int = 0, batch_size: int = 100):
        # load precomputed reference
        self.curriculum = Curriculum(phase_type=phase)
        wrapped = seed % self.curriculum.seeds_per_phase
        self._base_seed = wrapped
        self._batch_counter = 0

        ref = self.curriculum.get_reference(0, wrapped)
        self._encoder = ref["encoder"]
        self._mse = ref["mse"]
        self._complexity = len(ref["config"]["encoder"])
        self.batch_size = batch_size

        # record dims from the reference config
        ref_cfg = ref["config"]
        self._input_dim = ref_cfg["input_dim"]
        self._output_dim = ref_cfg["encoder"][-1]  # latent dimension

    def sample_batch(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        # advance seed each call to get fresh data
        seed = self._base_seed + self._batch_counter
        self._batch_counter += 1

        # _generate_data returns a single array X of shape (n_samples, input_dim)
        X_full = self.curriculum._generate_data(bs, seed)
        split = int(0.8 * len(X_full))
        Xtr, Xte = X_full[:split], X_full[split:]
        # autoencoder uses only X; Y_train/Y_val = None placeholders
        return (Xtr, None), (Xte, None)

    def reference_output(self, X_val):
        # Use the pretrained encoder to get “ground truth” latents
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
        sampler_fn(batch_size) -> dict mapping var names to np.arrays
        input_vars: list of keys for X
        target_vars: list of keys for Y
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
        # return the Y_val that was sampled alongside X_val
        return self._last_val_Y

    @property
    def reference_mse(self) -> float:
        # use the variance of Y as a normalization, for example
        return float(np.var(self._last_val_Y))

    def reference_complexity(self) -> float:
        # simple constant, or user‐defined metric
        return float(len(self.input_vars) + len(self.target_vars))

    @property
    def input_dim(self) -> int:
        # number of input features
        return len(self.input_vars)

    @property
    def output_dim(self) -> int:
        # number of target features
        return len(self.target_vars)
