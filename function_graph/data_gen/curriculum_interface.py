# data_gen/curriculum.py
from abc import ABC, abstractmethod
from typing import Iterator, Callable, Optional
from data_gen.problems import Problem


class CurriculumInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        problem_generator: Callable[[], Iterator[Problem]],
        increase_fn: Optional[Callable[['CurriculumInterface'], None]] = None,
        decrease_fn: Optional[Callable[['CurriculumInterface'], None]] = None
    ):
        """
        Initialize a curriculum with user-supplied logic.

        Args:
            problem_generator: Callable returning an iterator of Problem instances.
            increase_fn: Optional callable to adjust difficulty (invoked by increase_difficulty).
            decrease_fn: Optional callable to adjust difficulty (invoked by decrease_difficulty).
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Problem]:
        """
        Yields problems one by one from the provided generator.
        """
        pass

    def increase_difficulty(self) -> None:
        """
        Invoke the user-supplied increase function to raise difficulty, if any.
        """
        raise NotImplementedError

    def decrease_difficulty(self) -> None:
        """
        Invoke the user-supplied decrease function to lower difficulty, if any.
        """
        raise NotImplementedError
