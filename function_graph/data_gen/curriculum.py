# Create new file: /Users/tandava/auto_scientist/function_graph/data_gen/curriculum.py
from typing import Iterator, Callable, List, Type
from data_gen.problems import Problem
from data_gen.curriculum_interface import CurriculumInterface

class Curriculum(CurriculumInterface):
    def __init__(self, problem_generator: Callable[[], Iterator[Problem]]):
        """
        Manages a curriculum using a single problem generator.
        The generator should be a callable that returns an iterator over Problem instances.
        """
        self.problem_generator = problem_generator

    def __iter__(self) -> Iterator[Problem]:
        """
        Delegates iteration to the provided problem generator.
        The generator is expected to internally handle phase and seed iteration.
        """
        return self.problem_generator()

    @classmethod
    def seeded_problem_variations(cls, problem: Type["Problem"], phase: int, num: int):
        """
        Yields a sequence of problem instances for a given problem class.
        """
        for problem_seed in range(num):
            yield problem(phase=phase, problem_seed=problem_seed)