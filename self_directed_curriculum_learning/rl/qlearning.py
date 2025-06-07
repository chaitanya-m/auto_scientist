"""q_learning.py

A minimal **tabular Q‑learning** implementation that plugs into the generic
interfaces defined in *interfaces.py*.  The module is intentionally lightweight
so you can read it top‑to‑bottom:

* ``TabularQ``         – concrete :class:`interfaces.ValueFunction` storing a   
  nested ``dict`` of Q‑values and exposing a helper ``update_single``.
* ``EpsilonGreedyPolicy`` – concrete :class:`interfaces.Policy` that samples
  using ε‑greedy over the Q‑table.
* ``GymExperienceSource`` – simplest possible rollout generator (unused in the
  core learning loop here, but handy if you want to build a replay buffer).
* ``QLearningAgent``   – orchestrates the update rule inside
  ``learn_episode`` and provides a quick ``evaluate`` helper.

This implementation now **discretizes** continuous states (e.g.
CartPole observations) into fixed buckets so that tabular Q‑learning can
generalize.  If you provide an environment with discrete (already integer)
observations (e.g. FrozenLake), it simply casts to a hashable key.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple, Union
from utils import hashable_state, Discretizer, Experience

import numpy as np

from interfaces import (
    ActionT,
    PolicyFunction,
    StateT,
    Transition,
    ValueFunction,
    ExperienceGenerator,
)




# -----------------------------------------------------------------------------
# 1.  Q‑table value function
# -----------------------------------------------------------------------------
class TabularQ(ValueFunction[Any, ActionT]):
    """Simple (state, action) → value table with in‑place TD updates."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, init: float = 0.0):
        self._table: DefaultDict[Any, Dict[ActionT, float]] = defaultdict(
            lambda: defaultdict(lambda: init)
        )
        self.alpha = alpha
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Value queries
    # ------------------------------------------------------------------
    def v(self, state: Any) -> float:  # noqa: D401  – V(s) as max‑Q
        key = hashable_state(state)
        qs = self._table[key]
        return max(qs.values()) if qs else 0.0

    def q(self, state: Any, action: ActionT) -> float:  # noqa: D401
        key = hashable_state(state)
        return self._table[key][action]

    # ------------------------------------------------------------------
    # Learning helpers
    # ------------------------------------------------------------------
    def update(self, predictions: Sequence[float], targets: Sequence[float]) -> None:  # noqa: D401
        # Bulk updating rarely used in tabular—keep no-op.
        pass

    def update_single(self, state: Any, action: ActionT, target: float) -> None:
        key = hashable_state(state)
        q_old = self._table[key][action]
        self._table[key][action] = q_old + self.alpha * (target - q_old)


# -----------------------------------------------------------------------------
# 2.  ε‑greedy policy
# -----------------------------------------------------------------------------
class EpsilonGreedy(PolicyFunction[Any, ActionT]):
    """Samples ε‑greedy from a :class:`TabularQ`."""

    stochastic = True

    def __init__(
        self,
        q_table: TabularQ,
        actions: Sequence[ActionT],
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        discretizer: Discretizer | None = None,
    ) -> None:
        """
        :param eps_start: initial exploration rate
        :param eps_end: minimum exploration rate
        :param eps_decay: multiplicative decay per sample
        """
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.q_table = q_table
        self.actions = list(actions)
        self.discretizer = discretizer

    def sample(self, state: Any) -> ActionT:
        """Pick an action, then decay ε."""
        # discretize for lookup
        s = (
            self.discretizer(state)
            if self.discretizer and isinstance(state, np.ndarray)
            else state
        )
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self._greedy_action(s)
        # decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        return action

    def action(self, state: Any, *, deterministic: bool | None = None) -> ActionT:
        if deterministic:
            return self._greedy_action(
                self.discretizer(state) if self.discretizer and isinstance(state, np.ndarray) else state
            )
        return self.sample(state)

    def _greedy_action(self, state: Any) -> ActionT:
        key = hashable_state(state)
        qs = self.q_table._table[key]
        if not qs:  # unseen state – act uniformly at random
            return random.choice(self.actions)
        return max(qs.items(), key=lambda kv: kv[1])[0]

    # Fixed‑policy; gradients don’t apply.
    def update(self, gradients: Any) -> None:  # noqa: D401
        pass





# -----------------------------------------------------------------------------
# 3.  Agent orchestrator
# -----------------------------------------------------------------------------
class QLearningAgent:
    """Glue class that holds the table & policy and runs the TD‑update."""

    def __init__(
        self,
        env: Any,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        bins: Union[int, Sequence[int]] = 10,
        clip_low: Sequence[float] | float = -4.8,
        clip_high: Sequence[float] | float = 4.8,
    ) -> None:
        if not hasattr(env.action_space, "n"):
            raise TypeError("QLearningAgent requires a discrete action space.")
        self.env = env
        self.actions = list(range(env.action_space.n))
        self.gamma = gamma

        # Build discretizer over continuous space
        obs_space = env.observation_space
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            # 1) base env bounds
            obs_low = np.array(obs_space.low, dtype=float)
            obs_high = np.array(obs_space.high, dtype=float)
            # 2) hard-clip the unbounded velocity dims
            obs_low[1], obs_high[1] = -3.0,  3.0   # cart velocity
            obs_low[3], obs_high[3] = -3.5,  3.5   # pole angular velocity
            # 3) if user passed per-dim overrides, convert them
            if not isinstance(clip_low, (int, float)):
                clip_low_arr = np.array(clip_low, dtype=float)
            else:
                clip_low_arr = obs_low
            if not isinstance(clip_high, (int, float)):
                clip_high_arr = np.array(clip_high, dtype=float)
            else:
                clip_high_arr = obs_high

            # **Pass low & high positionally** then named args
            self.discretizer = Discretizer(
                obs_low,
                obs_high,
                bins=bins,
                clip_low=clip_low_arr,
                clip_high=clip_high_arr,
            )
        else:
            self.discretizer = None

        self.q_table = TabularQ(alpha=alpha, gamma=gamma)
        self.policy = EpsilonGreedy(
            self.q_table, self.actions, epsilon, discretizer=self.discretizer
        )

    # ------------------------------------------------------------------
    def learn_episode(self, max_steps: int | None = None) -> float:  # noqa: D401
        obs = self.env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0.0
        steps = 0
        while True:
            action = self.policy(state)   # this already discretizes for sampling

            # Discretize current state for the Q-update
            if self.discretizer is not None and isinstance(state, np.ndarray):
                s_key = self.discretizer(state)
            else:
                s_key = state

            step = self.env.step(action)
            if len(step) == 5:
                next_state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                next_state, reward, done, _ = step

            # Discretize next state for the TD target
            if self.discretizer is not None and isinstance(next_state, np.ndarray):
                s_next_key = self.discretizer(next_state)
            else:
                s_next_key = next_state

            # TD target: r + γ max_a' Q(s', a')
            best_next = self.q_table.v(s_next_key)
            target = reward + (0.0 if done else self.gamma * best_next)
            self.q_table.update_single(s_key, action, target)

            total_reward += reward
            state = next_state
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break
        return total_reward

    # ------------------------------------------------------------------
    def evaluate(self, episodes: int = 5) -> float:  # noqa: D401 – return mean reward
        # zero ε for pure greedy evaluation
        eps_backup = self.policy.epsilon
        self.policy.epsilon = 0.0
        scores: List[float] = []

        for _ in range(episodes):
            obs = self.env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            episode_reward = 0.0
            done = False
            while not done:
                action = self.policy(state, deterministic=True)
                step = self.env.step(action)
                if len(step) == 5:
                    state, r, term, trunc, _ = step
                    done = term or trunc
                else:
                    state, r, done, _ = step
                episode_reward += r
            scores.append(episode_reward)
        # restore ε
        self.policy.epsilon = eps_backup
        return float(np.mean(scores))


__all__ = [
    "TabularQ",
    "EpsilonGreedy",
    "GymExperienceSource",
    "QLearningAgent",
]


if __name__ == "__main__":
    # Simple test to ensure the module works as expected
    import gym

    env = gym.make("CartPole-v1")
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

    rewards: List[float] = []
    evaluations = []
    for episode in range(3000):
        total_reward = agent.learn_episode(max_steps=env._max_episode_steps)
        rewards.append(total_reward)

        #print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        # every 1000 episodes, print the average reward of that block
        freq = 250
        if (episode + 1) % freq == 0:
            block = rewards[-freq:]
            avg = sum(block) / len(block)
            start = episode - freq + 1
            end = episode + 1
            evaluations.append(agent.evaluate(episodes=100))

            print(f"Average reward for episodes {start}–{end}: {avg:.2f}")
   
    # Print all the evaluations at the end
    print("Final evaluations:", evaluations)
    env.close()

