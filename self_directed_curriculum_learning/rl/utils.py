from typing import Any, List, Tuple, Union, Sequence
import numpy as np
from interfaces import StateT, ActionT, PolicyFunction, Transition, ExperienceGenerator, BehaviorPolicy

# -----------------------------------------------------------------------------
# 0.  State discretizer for continuous spaces
# -----------------------------------------------------------------------------
class Discretizer:
    """Maps continuous observations to discrete buckets for tabular Q-learning."""

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        bins: int = 10,
        clip_low: Union[np.ndarray, float] = -4.8,
        clip_high: Union[np.ndarray, float] = 4.8,
    ) -> None:
        # Accept per-dimension or scalar clip bounds
        self.bins = bins
        self.edges: List[np.ndarray] = []

        # Ensure clip_low and clip_high are arrays of same shape as low/high
        clip_low_arr = (
            np.full_like(low, float(clip_low))
            if isinstance(clip_low, (int, float))
            else np.array(clip_low, dtype=float)
        )
        clip_high_arr = (
            np.full_like(high, float(clip_high))
            if isinstance(clip_high, (int, float))
            else np.array(clip_high, dtype=float)
        )

        for i in range(len(low)):
            lo = max(low[i], clip_low_arr[i])
            hi = min(high[i], clip_high_arr[i])
            if lo == -np.inf or hi == np.inf or lo >= hi:
                # Fallback to symmetric range if env gives infinite bounds
                delta = 1.0
                lo = -delta
                hi = delta
            # Create bin edges, excluding endpoints
            self.edges.append(np.linspace(lo, hi, bins + 1)[1:-1])

    def __call__(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert a continuous state into a tuple of bucket indices."""
        buckets: List[int] = []
        for i, edges in enumerate(self.edges):
            # Digitize returns 0..bins, but clamp to 0..(bins-1)
            idx = int(np.digitize(state[i], edges))
            if idx < 0:
                idx = 0
            elif idx >= self.bins:
                idx = self.bins - 1
            buckets.append(idx)
        return tuple(buckets)


def hashable_state(state: StateT) -> Any:
    """Ensure any state—discrete tuple or continuous array—is hashable."""
    if isinstance(state, tuple):
        return state
    if isinstance(state, np.ndarray):
        return tuple(state.ravel())
    return state


def evaluate_policy(
    policy: BehaviorPolicy, 
    env: Any, 
    episodes: int = 5
) -> float:
    """Evaluate a policy over multiple episodes."""
    scores = []
    
    for _ in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy.action(state)
            step = env.step(action)
            if len(step) == 5:
                state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                state, reward, done, _ = step
            episode_reward += reward
            
        scores.append(episode_reward)
        
    return float(np.mean(scores))


# -----------------------------------------------------------------------------
# 1. Experience source for Gym environments (implements ExperienceGenerator)
# -----------------------------------------------------------------------------
class Experience(ExperienceGenerator[Any, ActionT]):
    """Collects full transitions from an OpenAI Gym / Gymnasium env."""

    def collect(self, policy: PolicyFunction, env: Any, n_steps: int = 1, discretizer: Any = None) -> Tuple[List[Transition], float]:
        """
        Unified collection method.
        
        Args:
            n_steps: -1 = full episode (Monte Carlo), n > 0 = exactly n steps (1 = Temporal Difference)
            discretizer: Optional state discretizer
        
        Returns:
            (transitions, total_reward)
        """
        if n_steps == -1:
            # Full episode collection
            return self.collect_episode(policy, env, max_steps=None, discretizer=discretizer)
        else:
            # N-step collection with reward tracking
            transitions = []
            total_reward = 0.0
            
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            
            for _ in range(n_steps):
                action = policy(state)
                step = env.step(action)
                if len(step) == 5:
                    next_state, reward, term, trunc, _ = step
                    done = term or trunc
                else:
                    next_state, reward, done, _ = step

                # Discretize states if discretizer provided
                s_key = discretizer(state) if discretizer else state
                s_next_key = discretizer(next_state) if discretizer else next_state
                
                transitions.append(Transition(
                    state=s_key,
                    action=action,
                    reward=reward,
                    next_state=s_next_key,
                    done=done
                ))

                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            return transitions, total_reward

    def collect_episode(
        self, 
        policy: PolicyFunction, 
        env: Any, 
        max_steps: int | None = None,
        discretizer: Any = None
    ) -> Tuple[List[Transition], float]:
        """Collect a full episode of experience."""
        transitions: List[Transition] = []
        total_reward = 0.0
        
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        steps = 0
        
        while True:
            action = policy(state)
            
            step = env.step(action)
            if len(step) == 5:
                next_state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                next_state, reward, done, _ = step

            # Discretize states if discretizer provided
            s_key = discretizer(state) if discretizer else state
            s_next_key = discretizer(next_state) if discretizer else next_state
            
            transitions.append(Transition(
                state=s_key,
                action=action,
                reward=reward,
                next_state=s_next_key,
                done=done
            ))

            total_reward += reward
            state = next_state
            steps += 1
            
            if done or (max_steps is not None and steps >= max_steps):
                break
        
        return transitions, total_reward

def create_discretizer(
    env: Any,
    bins: Union[int, Sequence[int]],
    clip_low: Sequence[float] | float,
    clip_high: Sequence[float] | float
) -> Discretizer | None:
    """Helper to create discretizer for continuous observation spaces."""
    obs_space = env.observation_space
    if not (hasattr(obs_space, "low") and hasattr(obs_space, "high")):
        return None
        
    obs_low = np.array(obs_space.low, dtype=float)
    obs_high = np.array(obs_space.high, dtype=float)
    
    # Handle infinite bounds by using clip values
    if not isinstance(clip_low, (int, float)):
        clip_low_arr = np.array(clip_low, dtype=float)
    else:
        clip_low_arr = np.full_like(obs_low, float(clip_low))
        
    if not isinstance(clip_high, (int, float)):
        clip_high_arr = np.array(clip_high, dtype=float)
    else:
        clip_high_arr = np.full_like(obs_high, float(clip_high))

    # Replace infinite bounds with clip values
    mask_low_inf = np.isinf(obs_low)
    mask_high_inf = np.isinf(obs_high)
    
    obs_low = np.where(mask_low_inf, clip_low_arr, obs_low)
    obs_high = np.where(mask_high_inf, clip_high_arr, obs_high)

    return Discretizer(
        obs_low, obs_high,
        bins=bins,
        clip_low=clip_low_arr,
        clip_high=clip_high_arr,
    )
