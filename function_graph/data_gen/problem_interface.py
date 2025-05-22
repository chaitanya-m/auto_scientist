from abc import ABC, abstractmethod

class Problem(ABC):
    """
    Abstract interface for a ground‐truth task.

    Implementations must provide:
      - sample_batch(): draw a batch of data (train & val)
      - reference_output(): return the true outputs for a batch
      - reference_mse: the “target” error scale (for normalization)
      - reference_complexity(): a scalar measuring ground‐truth complexity
      - input_dim / output_dim: dimensionalities of X and Y
      - problem_generator: yields a sequence of problems for a given phase
    """
    @abstractmethod
    def sample_batch(self, batch_size: int):
        pass

    @abstractmethod
    def reference_output(self, X_val):
        pass

    @property
    @abstractmethod
    def reference_mse(self) -> float:
        pass

    @abstractmethod
    def reference_complexity(self) -> float:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass
