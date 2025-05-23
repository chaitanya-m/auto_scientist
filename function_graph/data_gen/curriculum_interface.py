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

