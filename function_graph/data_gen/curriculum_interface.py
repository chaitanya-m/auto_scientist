# data_gen/curriculum.py
from abc import ABC, abstractmethod
from typing import Iterator, Callable, List, Type
from data_gen.problems import Problem


class CurriculumInterface(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Problem]:
        """
        Yields problems one by one.
        """
        pass

    @classmethod
    @abstractmethod
    def seeded_problem_variations(cls, problem: Type["Problem"], phase: str, num: int) -> Iterator["Problem"]:
        """
        Yields a sequence of problem instances for a given problem class.
        """
        pass

