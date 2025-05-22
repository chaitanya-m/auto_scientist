# Create new file: /Users/tandava/auto_scientist/function_graph/data_gen/curriculum.py
from typing import Iterator, Callable, List, Type
from data_gen.problems import Problem
from data_gen.curriculum_interface import CurriculumInterface

class Curriculum(CurriculumInterface):
    def __init__(self, problem_generators: List[Callable[[str, int], Problem]]):
        """
        Manages a curriculum as a sequence of problem generators.
        Each generator should be a callable accepting (phase, problem_seed) and returning a Problem instance.
        """
        self.problem_generators = problem_generators

    def __iter__(self) -> Iterator[Problem]:
        """
        Yields problems one by one.
        """
        for generator in self.problem_generators:
            yield generator()

    @classmethod
    def seeded_problem_variations(cls, problem: Type["Problem"], phase: str, num: int):
        """
        Yields a sequence of problem instances for a given problem class.
        """
        for problem_seed in range(num):
            yield problem(phase=phase, problem_seed=problem_seed)