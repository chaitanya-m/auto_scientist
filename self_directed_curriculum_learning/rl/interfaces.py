"""interfaces.py

A modular, orthogonal set of **abstract interfaces** covering the building
blocks needed to implement the overwhelming majority of textbook and modern
reinforcement‑learning algorithms: from tabular Q‑learning to PPO, DDPG, SAC,
IMPALA, and beyond.

Design principles
-----------------
* **Narrow contracts** – each class handles one concern.
* **Composability** – algorithms mix concrete implementations freely.
* **Stateless call‑sites** – pass data in, get data out; side‑effects live only
  in `update/step` methods so that vectorised / distributed setups are simple.
* **Type hints** – generic type‑vars keep the file framework‑agnostic yet help
  static analysis (Pyright, MyPy, IDE autocompletion).

Concrete subclasses *may* add convenience helpers (save/load, device transfer,
`@torch.no_grad`, etc.); the *contracts* below should stay minimal.
"""

from __future__ import annotations

import abc
import math
from typing import (
    Any,
    Generic,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Protocol,
)

# -----------------------------------------------------------------------------
# 1.  Generic type aliases
# -----------------------------------------------------------------------------
StateT  = TypeVar("StateT",  covariant=True)
ActionT = TypeVar("ActionT", covariant=True)
RewardT = float  # scalar reward – libs can override with Tensor alias


class Transition(NamedTuple, Generic[StateT, ActionT]):
    """A single interaction slice (s, a, r, s′, done[, info])."""

    state: StateT
    action: ActionT
    reward: RewardT
    next_state: StateT
    done: bool
    info: Optional[Mapping[str, Any]] = None


# -----------------------------------------------------------------------------
# 2.  Policy function
# -----------------------------------------------------------------------------
class PolicyFunction(abc.ABC, Generic[StateT, ActionT]):
    """Maps states to actions.  Supports deterministic & stochastic variants."""

    stochastic: bool = True  # subclasses override if deterministic

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def __call__(
        self, state: StateT, *, deterministic: bool | None = None
    ) -> ActionT:
        """Return an action for the given *state*.
        If *deterministic* is `True`, use a greedy action; if `False`, sample
        from the policy distribution.  If `None`, use the default policy behavior.
        """
        return self.action(state, deterministic=deterministic)

    @abc.abstractmethod
    def action(
        self, state: StateT, *, deterministic: bool | None = None
    ) -> ActionT:
        """Return an action – sample or greedy depending on *deterministic*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Probability evaluation (stochastic‑only)
    # ------------------------------------------------------------------
    def log_prob(self, state: StateT, action: ActionT) -> float:  # noqa: D401
        """log π(a∣s).  Raises if the policy is deterministic."""
        if not self.stochastic:
            raise NotImplementedError("log_prob undefined for deterministic policies.")
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def update(self, gradients: Any) -> None:
        """Apply *gradients* (framework‑specific tensor)."""
        raise NotImplementedError


class BehaviorPolicy(PolicyFunction[StateT, ActionT]):
    """Policy used to generate actions in the environment (behavior)."""
    # Inherit .action and .__call__ behavior
    pass


class TargetPolicy(PolicyFunction[StateT, ActionT]):
    """Policy used to select actions for computing TD targets (target)."""
    # Inherit .action and .__call__ behavior
    pass


# -----------------------------------------------------------------------------
# 3.  Value function
# -----------------------------------------------------------------------------
class ValueFunction(abc.ABC, Generic[StateT, ActionT]):
    """Either V(s) **or** Q(s,a).  Call‑site decides which variant."""

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def __call__(self, state: StateT, action: Optional[ActionT] = None) -> float:
        return self.q(state, action) if action is not None else self.v(state)  # type: ignore[arg-type]

    def v(self, state: StateT) -> float:  # noqa: D401
        """V(s).  Raises if the implementation is Q‑only."""
        raise NotImplementedError

    def q(self, state: StateT, action: ActionT) -> float:  # noqa: D401
        """Q(s,a).  Raises if the implementation is V‑only."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def update(
        self, predictions: Sequence[float], targets: Sequence[float]
    ) -> None:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 4.  Experience generator
# -----------------------------------------------------------------------------
class ExperienceGenerator(abc.ABC, Generic[StateT, ActionT]):
    """Produces experience through environment interactions.  Works as an *iterator*."""

    def __iter__(self) -> Iterator[Transition[StateT, ActionT]]:  # noqa: D401
        while True:
            yield from self.collect(n_steps=1)

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def collect(
        self,
        policy: PolicyFunction[StateT, ActionT],
        env: Any,
        n_steps: int = 1,
    ) -> Sequence[Transition[StateT, ActionT]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Replay‑buffer API (noop for on‑policy)
    # ------------------------------------------------------------------
    def store(self, transition: Transition[StateT, ActionT]) -> None:  # noqa: D401
        pass

    def sample_batch(
        self, batch_size: int
    ) -> Optional[Sequence[Transition[StateT, ActionT]]]:
        return None

    def clear(self) -> None:  # noqa: D401
        pass


# -----------------------------------------------------------------------------
# 5.  Return / advantage estimation
# -----------------------------------------------------------------------------
class ReturnEstimator(abc.ABC, Generic[StateT, ActionT]):
    """Compute Rₜ targets (and optionally advantages) for policy/value updates."""

    def __init__(self, discount: float = 0.99, gae_lambda: Optional[float] = None):
        self.discount = discount
        self.gae_lambda = gae_lambda

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def returns(
        self,
        transitions: Sequence[Transition[StateT, ActionT]],
        critic: ValueFunction[StateT, ActionT],
    ) -> Sequence[float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def advantages(
        self,
        transitions: Sequence[Transition[StateT, ActionT]],
        critic: ValueFunction[StateT, ActionT],
    ) -> Sequence[float]:
        v_pred = [critic.v(t.state) for t in transitions]  # type: ignore[misc]
        r_tar = self.returns(transitions, critic)
        return [r - v for r, v in zip(r_tar, v_pred)]


# -----------------------------------------------------------------------------
# 6.  Optimisers / updaters
# -----------------------------------------------------------------------------
class ActorUpdater(abc.ABC, Generic[StateT, ActionT]):
    """Consumes (s, a, Â) and updates the **policy** parameters."""

    @abc.abstractmethod
    def loss(
        self,
        states: Sequence[StateT],
        actions: Sequence[ActionT],
        advantages: Sequence[float],
        *,
        log_probs_old: Optional[Sequence[float]] = None,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, loss: float) -> Mapping[str, float]:
        """Apply gradients; return diagnostics (e.g. learning‑rate, grad‑norm)."""
        raise NotImplementedError


class CriticUpdater(abc.ABC, Generic[StateT, ActionT]):
    """Updates **value‑function** parameters."""

    @abc.abstractmethod
    def loss(self, predictions: Sequence[float], targets: Sequence[float]) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, loss: float) -> Mapping[str, float]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 7.  Exploration / behaviour‑target split
# -----------------------------------------------------------------------------
class PolicyEvaluator(abc.ABC, Generic[StateT, ActionT]):
    """Decouples behaviour‑policy (data gen) and target‑policy (bootstrapping)."""

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def behaviour_action(
        self, behaviour: PolicyFunction[StateT, ActionT], state: StateT
    ) -> ActionT:
        raise NotImplementedError

    @abc.abstractmethod
    def target_action(
        self, target: PolicyFunction[StateT, ActionT], state: StateT
    ) -> ActionT:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def importance_weight(
        self,
        behaviour_logp: float,
        target_logp: float,
    ) -> float:
        """Default: π/μ importance ratio; override for tricks like capping."""
        return math.exp(target_logp - behaviour_logp)


# -----------------------------------------------------------------------------
# 8.  Target‑network mix‑in (stability tricks)
# -----------------------------------------------------------------------------
class TargetNetworkMixin(Protocol):
    """Provides *hard* and *soft* parameter copying helpers."""

    def hard_update(self) -> None:  # noqa: D401
        ...

    def soft_update(self, tau: float) -> None:  # noqa: D401
        ...


# -----------------------------------------------------------------------------
# 9.  Entropy bonus helpers (SAC / PPO‑Entropy)
# -----------------------------------------------------------------------------
class EntropyBonus(abc.ABC, Generic[StateT, ActionT]):
    """Computes H[π(·|s)] and its (possibly learned) coefficient."""

    @abc.abstractmethod
    def entropy(self, state: StateT) -> float:  # noqa: D401
        raise NotImplementedError

    @abc.abstractmethod
    def coefficient(self) -> float:  # noqa: D401
        raise NotImplementedError
