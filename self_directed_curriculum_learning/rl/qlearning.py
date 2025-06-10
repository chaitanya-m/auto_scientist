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
from utils import hashable_state, Discretizer, Experience, evaluate_policy, create_discretizer

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
        self.episode_count = 0

    def sample(self, state: Any) -> ActionT:
        """Pick an action using epsilon-greedy strategy."""
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

    def end_episode(self) -> None:
        """Handle epsilon decay at end of episode."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


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

class EligibilityTraceCriticUpdater(CriticUpdater[Any, ActionT]):
    """
    Applies eligibility traces for temporal credit assignment.
    
    λ = 1: All visited states in batch updated equally. 
        If the update frequency is a full episode (set to -1 in this code), this is equivalent to Monte Carlo, 
        i.e. all states in the the entire episode trajectory prior to a state get updated with equal credit.
    
    λ = 0: Only the most recent state-action pair in the batch is updated. this is equivalent to temporal-difference learning. 

    0 < λ < 1: Exponentially decaying credit assignment to states depending on how long ago they were visited in the trajectory.
    """
    
    def __init__(self, q_table: TabularQ, alpha: float = 0.1, gamma: float = 0.99, lambda_: float = 0.9):
        self.q_table = q_table
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.traces: DefaultDict[Any, Dict[ActionT, float]] = defaultdict(lambda: defaultdict(float))
        self._episode_active = False
    
    def loss(self, predictions: Sequence[float], targets: Sequence[float]) -> float:
        """Compute MSE loss for diagnostics."""
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

    def step(self, experiences: Sequence[Transition[Any, ActionT]]) -> Mapping[str, float]:
        """Apply eligibility trace updates to the Q-table."""
        
        # Clear traces at episode start
        if not self._episode_active:
            self.traces.clear()
            self._episode_active = True
        
        total_td_error = 0.0
        updates_made = 0
        
        for exp in experiences:
            s_key = exp.state
            s_next_key = exp.next_state
            
            # Compute TD error (same as standard Q-learning)
            current_q = self.q_table.q(s_key, exp.action)
            best_next = self.q_table.v(s_next_key) if not exp.done else 0.0
            target = exp.reward + self.gamma * best_next
            td_error = target - current_q
            
            total_td_error += abs(td_error)
            
            # FIRST: Update all existing traces with TD error
            states_to_remove = []
            for state in self.traces:
                actions_to_remove = []
                for action in self.traces[state]:
                    trace = self.traces[state][action]
                    
                    if trace > 0.01:  # Only update significant traces
                        # Apply proportional update
                        old_q = self.q_table.q(state, action)
                        new_q = old_q + self.alpha * td_error * trace
                        self.q_table._table[state][action] = new_q
                        updates_made += 1
                    
                    # Decay trace
                    self.traces[state][action] *= self.gamma * self.lambda_
                    
                    # Mark small traces for removal
                    if self.traces[state][action] < 0.01:
                        actions_to_remove.append(action)
                
                # Remove small traces
                for action in actions_to_remove:
                    del self.traces[state][action]
                
                # Mark empty states for removal
                if not self.traces[state]:
                    states_to_remove.append(state)
        
            # Remove empty states
            for state in states_to_remove:
                del self.traces[state]
        
            # THEN: Set eligibility trace for current state-action to 1.0 (replacing traces)
            self.traces[s_key][exp.action] = 1.0
            
            # Also update the current state-action with the TD error
            old_q = self.q_table.q(s_key, exp.action)
            new_q = old_q + self.alpha * td_error * 1.0
            self.q_table._table[s_key][exp.action] = new_q
            updates_made += 1
            
            # Clear traces on episode end
            if exp.done:
                self.traces.clear()
                self._episode_active = False
    
        avg_td_error = total_td_error / len(experiences) if experiences else 0.0
        active_traces = sum(len(state_traces) for state_traces in self.traces.values())
        
        return {
            "td_error": avg_td_error,
            "active_traces": active_traces,
            "trace_updates": updates_made
        }
# -----------------------------------------------------------------------------
# 3.  Agent orchestrator
# -----------------------------------------------------------------------------
class QLearningAgent:
    """Simple orchestrator that delegates to interface implementations."""

    def __init__(
        self,
        env: Any,
        target_policy: TargetPolicy[Any, ActionT],
        behavior_policy: BehaviorPolicy[Any, ActionT],
        critic_updater: CriticUpdater[Any, ActionT],
        experience_generator: Any,  # Experience from utils.py
        discretizer: Discretizer | None = None,
    ) -> None:
        """Pure dependency injection constructor."""
        self.env = env
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.critic_updater = critic_updater
        self.experience_generator = experience_generator
        self.discretizer = discretizer

    def learn(self, max_steps: int | None = None, update_frequency: int = -1) -> float:
        """Learn with configurable update frequency."""
        
        # Collect experiences based on update frequency
        experiences, total_reward = self.experience_generator.collect(
            self.target_policy.sample, 
            self.env, 
            n_steps=update_frequency,
            discretizer=self.discretizer
        )
        
        # Update critic
        self.critic_updater.step(experiences)
        
        return total_reward


# Factory function for easy setup
def create_qlearning_agent(
    env: Any,
    *,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    bins: Union[int, Sequence[int]] = 10,
    clip_low: Sequence[float] | float = -4.8,
    clip_high: Sequence[float] | float = 4.8,
) -> QLearningAgent:
    """Factory to create a complete Q-learning setup with all dependencies."""
    if not hasattr(env.action_space, "n"):
        raise TypeError("QLearningAgent requires a discrete action space.")
    
    actions = list(range(env.action_space.n))
    
    # Create discretizer with environment-appropriate bounds
    # For CartPole, use velocity-specific bounds
    if hasattr(env, 'spec') and env.spec and 'CartPole' in env.spec.id:
        # CartPole-specific bounds for unbounded velocity dimensions
        clip_low = [-4.8, -3.0, -0.5, -3.5]  # pos, vel, angle, angular_vel
        clip_high = [4.8, 3.0, 0.5, 3.5]
    
    discretizer = create_discretizer(env, bins, clip_low, clip_high)
    
    # Create components
    q_table = TabularQ(alpha=alpha, gamma=gamma)
    target_policy = EpsilonGreedy(
        q_table=q_table,
        actions=actions,
        eps_start=epsilon,
        eps_end=0.01,
        eps_decay=0.9995,  # Back to episode-based rate
        discretizer=discretizer,
    )
    behavior_policy = DeterministicPolicy(target_policy)
    critic_updater = TabularCriticUpdater(q_table, alpha=alpha)
    experience_generator = Experience()
    
    return QLearningAgent(
        env=env,
        target_policy=target_policy,
        behavior_policy=behavior_policy,
        critic_updater=critic_updater,
        experience_generator=experience_generator,
        discretizer=discretizer,
    )




if __name__ == "__main__":
    # Simple test to ensure the module works as expected
    import gym

    env = gym.make("CartPole-v1")
    agent = create_qlearning_agent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

    rewards: List[float] = []
    evaluations = []
    for episode in range(3000):
        # Use the new learn method with episode-based updates (default)
        total_reward = agent.learn(max_steps=env._max_episode_steps, update_frequency=-1)
        rewards.append(total_reward)

        #print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        # every 1000 episodes, print the average reward of that block
        freq = 250
        if (episode + 1) % freq == 0:
            block = rewards[-freq:]
            avg = sum(block) / len(block)
            start = episode - freq + 1
            end = episode + 1
            evaluations.append(evaluate_policy(agent.behavior_policy, env, episodes=100))

            print(f"Average reward for episodes {start}–{end}: {avg:.2f}")
   
    # Print all the evaluations at the end
    print("Final evaluations:", evaluations)
    evaluation_score = evaluate_policy(agent.behavior_policy, env, episodes=100)
    print("Evaluation score:", evaluation_score)
    env.close()
