from abc import ABC, abstractmethod
from typing import Iterator

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

    @classmethod
    @abstractmethod
    def get_phase_config(cls, phase: int) -> dict:
        """
        Returns the configuration dictionary for the given numeric phase.
        Phase 0 is basic; phase 1 is intermediate; phases > 1 build on intermediate.
        """
        pass

    @classmethod
    @abstractmethod
    def seeded_problem_variations(cls, phase: int, num: int) -> Iterator["Problem"]:
        """
        Yields a sequence of problem instances for the given phase.

        Arguments:
           phase: An integer representing the problem phase.
           num  : Number of problems (different seeds) to generate.
        """
        pass
