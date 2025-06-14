from typing import Iterator, Callable, Optional, Type
from itertools import count
from curriculum_generator.problems import Problem
from curriculum_generator.curriculum_interface import CurriculumInterface


class AutoEncoderCurriculum(CurriculumInterface):
    """
    A Curriculum for generating problems with adjustable difficulty.
    The implementor must provide a problem generator and optional difficulty adjustment functions.
    """
    @property
    def difficulty(self) -> Optional[int]:
        """Current difficulty level of this curriculum, if available."""
        holder = getattr(self, '_difficulty_holder', None)
        if holder is not None:
            return holder.get('difficulty')
        return None

    @property
    def num_problems(self) -> Optional[int]:
        """Number of problems per batch; None indicates infinite."""
        return self._num_problems

    def __init__(
        self,
        problem_generator: Callable[[], Iterator[Problem]],
        increase_fn: Optional[Callable[['AutoEncoderCurriculum'], None]] = None,
        decrease_fn: Optional[Callable[['AutoEncoderCurriculum'], None]] = None,
        maintain_fn: Optional[Callable[['AutoEncoderCurriculum'], None]] = None,
        num_problems: Optional[int] = None
    ):
        """
        Initialize a curriculum with a problem generator and optional difficulty logic.

        Args:
            problem_generator: Callable that returns an iterator of Problem instances.
            increase_fn: Optional function to adjust difficulty when increase_difficulty is called.
            decrease_fn: Optional function to adjust difficulty when decrease_difficulty is called.
            maintain_fn: Optional function to maintain difficulty when maintain_difficulty is called.
            num_problems: If finite, indicates a one-time batch; None enables infinite adjustment.

        Raises:
            ValueError: if num_problems is finite and any difficulty handler is provided.
        """
        # (we allow hooks on finite or infinite streams;
        #  to disable adjustment simply pass increase_fn/decrease_fn=None)

        self._generator = problem_generator
        self._increase_fn = increase_fn
        self._decrease_fn = decrease_fn
        self._maintain_fn = maintain_fn
        self._num_problems = num_problems

    def __iter__(self) -> Iterator[Problem]:
        """
        Yield problems from the provided generator.
        """
        return self._generator()

    def increase_difficulty(self) -> None:
        """
        Invoke the user-supplied increase function, if provided.

        Raises:
            NotImplementedError: if no increase function was given.
        """
        if not self._increase_fn:
            raise NotImplementedError("This curriculum cannot increase difficulty.")
        self._increase_fn(self)

    def decrease_difficulty(self) -> None:
        """
        Invoke the user-supplied decrease function, if provided.

        Raises:
            NotImplementedError: if no decrease function was given.
        """
        if not self._decrease_fn:
            raise NotImplementedError("This curriculum cannot decrease difficulty.")
        self._decrease_fn(self)

    def maintain_difficulty(self) -> None:
        """
        Invoke the user-supplied maintain function (no-op by default).

        Raises:
            NotImplementedError: if no maintain_fn was given.
        """
        if not self._maintain_fn:
            raise NotImplementedError("This curriculum cannot maintain difficulty.")
        self._maintain_fn(self)

    @classmethod
    def default(
        cls,
        problem_cls: Type[Problem],
        initial_difficulty: int,
        num_problems: Optional[int] = 2
    ) -> 'AutoEncoderCurriculum':
        """
        Factory for a default Curriculum instance using a Problem subclass.
        - If num_problems is finite, yields that finite batch once, with no adjustment handlers.
        - If num_problems is None, yields an infinite stream and enables difficulty adjustment.

        Args:
            problem_cls: A Problem subclass implementing seeded_problem_variations.
            initial_difficulty: The starting difficulty for problem generation.
            num_problems: Number of problems (seeds) per iteration; None for infinite.

        Returns:
            A Curriculum object configured for the default behavior.
        """
        # Prepare mutable difficulty holder
        holder = {'difficulty': initial_difficulty}

        # Define generator based on finite vs infinite
        if num_problems is None:
            def gen() -> Iterator[Problem]:
                # infinite stream: one problem at a time
                for _ in count():
                    yield from problem_cls.seeded_problem_variations(holder['difficulty'], 1)
        else:
            def gen() -> Iterator[Problem]:
                # finite batch
                return iter(problem_cls.seeded_problem_variations(holder['difficulty'], num_problems))

        # Always wire up a difficulty holder and the three hooks.
        def inc(cur: 'AutoEncoderCurriculum') -> None:
            holder['difficulty'] += 1

        def dec(cur: 'AutoEncoderCurriculum') -> None:
            holder['difficulty'] = max(0, holder['difficulty'] - 1)

        def keep(cur: 'AutoEncoderCurriculum') -> None:
            pass

        curr = cls(
            problem_generator=gen,
            increase_fn=inc,
            decrease_fn=dec,
            maintain_fn=keep,
            num_problems=num_problems
        )
        curr._difficulty_holder = holder
        return curr
