# data_gen/curriculum.py
import numpy as np
from typing import Iterator, Callable, List
from data_gen.problems import Problem


class Curriculum:
    def __init__(self, problem_generators: List[Callable[[], Problem]]):
        """
        Manages a curriculum as a sequence of problems.

        Args:
            problem_generators: A list of functions, each returning a Problem instance.
        """
        self.problem_generators = problem_generators

    def __iter__(self) -> Iterator[Problem]:
        """
        Yields problems one by one.
        """
        for generator in self.problem_generators:
            yield generator()

