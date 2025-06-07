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
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple, Union
from utils import hashable_state, Discretizer, Experience

import numpy as np

from interfaces import (
    ActionT,
    ValueFunction,
    TargetPolicy,
    BehaviorPolicy,
    CriticUpdater,
    Transition
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
# 2.  Target (policy to learn) and behavior (policy to use) policies
# -----------------------------------------------------------------------------
class EpsilonGreedy(TargetPolicy[Any, ActionT]):
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


# ---------------------------------------------------------------------
class DeterministicPolicy(BehaviorPolicy[Any, ActionT]):
    """Wrap EpsilonGreedy or other TargetPolicy to always act deterministically for off-policy control."""
    stochastic = False

    def __init__(self, base_policy: TargetPolicy[Any, ActionT]) -> None:
        self.base = base_policy

    def action(self, state: Any, *, deterministic: bool | None = None) -> ActionT:
        # force deterministic=True on the wrapped policy
        return self.base.action(state, deterministic=True)

    def update(self, gradients: Any) -> None:
        pass  # noqa: D401  # no-op, as no gradients apply to the q_table


# Add a CriticUpdater for tabular Q-learning
class TabularCriticUpdater(CriticUpdater[Any, ActionT]):
    """Updates TabularQ using MSE loss and direct table updates."""
    
    def __init__(self, q_table: TabularQ, alpha: float = 0.1):
        self.q_table = q_table
        self.alpha = alpha
    
    def loss(self, predictions: Sequence[float], targets: Sequence[float]) -> float:
        """Compute MSE loss for diagnostics."""
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    def step(self, experiences: Sequence[Transition[Any, ActionT]]) -> Mapping[str, float]:
        """Actually perform the Q-table updates."""
        predictions = []
        targets = []
        
        for exp in experiences:
            # Discretize states (this should be moved to agent level)
            s_key = exp.state  # assume already discretized
            s_next_key = exp.next_state
            
            # Current Q-value
            pred = self.q_table.q(s_key, exp.action)
            
            # TD target
            best_next = self.q_table.v(s_next_key) if not exp.done else 0.0
            target = exp.reward + 0.99 * best_next  # gamma should be passed in
            
            # Apply update
            self.q_table.update_single(s_key, exp.action, target)
            
            predictions.append(pred)
            targets.append(target)
        
        loss = self.loss(predictions, targets)
        return {"critic_loss": loss}

# -----------------------------------------------------------------------------
# 3.  Agent orchestrator
# -----------------------------------------------------------------------------
class QLearningAgent:
    """Simple orchestrator that delegates to interface implementations."""

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
        """Constructor that maintains backward compatibility with existing main function."""
        if not hasattr(env.action_space, "n"):
            raise TypeError("QLearningAgent requires a discrete action space.")
        
        self.env = env
        self.actions = list(range(env.action_space.n))
        self.gamma = gamma

        # Create discretizer if needed
        obs_space = env.observation_space
        self.discretizer = None
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            obs_low = np.array(obs_space.low, dtype=float)
            obs_high = np.array(obs_space.high, dtype=float)
            # Hard-clip the unbounded velocity dims for CartPole
            obs_low[1], obs_high[1] = -3.0, 3.0   # cart velocity
            obs_low[3], obs_high[3] = -3.5, 3.5   # pole angular velocity
            
            if not isinstance(clip_low, (int, float)):
                clip_low_arr = np.array(clip_low, dtype=float)
            else:
                clip_low_arr = obs_low
            if not isinstance(clip_high, (int, float)):
                clip_high_arr = np.array(clip_high, dtype=float)
            else:
                clip_high_arr = obs_high

            self.discretizer = Discretizer(
                obs_low, obs_high,
                bins=bins,
                clip_low=clip_low_arr,
                clip_high=clip_high_arr,
            )

        # Create components using dependency injection pattern
        self.q_table = TabularQ(alpha=alpha, gamma=gamma)
        self.target_policy = EpsilonGreedy(
            q_table=self.q_table,
            actions=self.actions,
            eps_start=epsilon,
            eps_end=0.01,
            eps_decay=0.9995,
            discretizer=self.discretizer,
        )
        self.behavior_policy = DeterministicPolicy(self.target_policy)
        self.critic_updater = TabularCriticUpdater(self.q_table, alpha=alpha)
        self.experience_generator = Experience()

    def collect_experience(self, n_steps: int = 1) -> Sequence[Transition[Any, ActionT]]:
        """Delegate to experience generator."""
        return self.experience_generator.collect(self.target_policy, self.env, n_steps)

    def update(self, experiences: Sequence[Transition[Any, ActionT]]) -> Mapping[str, float]:
        """Delegate to critic updater."""
        # Remove the double discretization - experiences are already discretized!
        return self.critic_updater.step(experiences)

    def learn_episode(self, max_steps: int | None = None) -> float:
        """Convenience method that maintains backward compatibility."""
        obs = self.env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0.0
        steps = 0
        experiences = []
        
        while True:
            action = self.target_policy.sample(state)
            
            step = self.env.step(action)
            if len(step) == 5:
                next_state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                next_state, reward, done, _ = step

            # Discretize states for the transition
            s_key = self.discretizer(state) if self.discretizer else state
            s_next_key = self.discretizer(next_state) if self.discretizer else next_state
            
            transition = Transition(
                state=s_key,  # Already discretized
                action=action,
                reward=reward,
                next_state=s_next_key,  # Already discretized
                done=done,
                info={}
            )
            experiences.append(transition)

            total_reward += reward
            state = next_state  # Keep raw state for next iteration
            steps += 1
            
            if done or (max_steps is not None and steps >= max_steps):
                break
    
        # Update with all experiences from this episode
        if experiences:
            self.update(experiences)  # Now passes correctly discretized experiences
    
        # Handle epsilon decay
        self.target_policy.epsilon = max(
            self.target_policy.eps_end, 
            self.target_policy.epsilon * self.target_policy.eps_decay
        )
        
        return total_reward

    def evaluate(self, episodes: int = 5) -> float:
        """Run evaluation episodes using behavior policy."""
        scores = []
        
        for _ in range(episodes):
            obs = self.env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.behavior_policy.action(state)
                step = self.env.step(action)
                if len(step) == 5:
                    state, reward, term, trunc, _ = step
                    done = term or trunc
                else:
                    state, reward, done, _ = step
                episode_reward += reward
                
            scores.append(episode_reward)
            
        return float(np.mean(scores))


__all__ = [
    "TabularQ",
    "EpsilonGreedy",
    "DeterministicPolicy", 
    "TabularCriticUpdater",
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

