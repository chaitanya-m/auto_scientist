# envs/environment.py

from typing import Tuple, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from utils.nn import create_minimal_graphmodel
from envs.data_types import State, AgentState

# -----------------------
# Environment Interface
# -----------------------

class Environment(ABC):
    def __init__(self, transition_rule, reward_rule, initial_state):
        """
        A minimal state transition environment interface.

        :param transition_rule: A callable that, given the current state and a dictionary of actions,
                                returns the next state.
        :param reward_rule: A callable that, given the current state, actions, and next state,
                            computes and returns a reward (or dictionary of rewards).
        """
        self.transition_rule = transition_rule
        self.reward_rule = reward_rule
        self.state = initial_state  # No initial state is generated automatically.
        self.initial_state = initial_state

    @abstractmethod
    def reset(self, transition_rule=None, reward_rule=None, state=None):
        """
        Resets the environment by updating its rules and state only if new values are provided.

        :param transition_rule: (Optional) A new transition rule; if None, preserves the current rule.
        :param reward_rule: (Optional) A new reward rule; if None, preserves the current rule.
        :param state: (Optional) A new state object; if None, goes back to the original initial state.
        :return: The state after reset.
        """
        if transition_rule is not None:
            self.transition_rule = transition_rule
        if reward_rule is not None:
            self.reward_rule = reward_rule
        
        self.state = state if state is not None else self.initial_state

        return self.state


    @abstractmethod
    def step(self, actions: Dict[Any, Any]) -> Tuple[Any, Dict[Any, float]]:
        """
        Applies the transition rule to update the state.

        :param actions: A dictionary mapping agent IDs to their actions.
        :return: A tuple (next_state, reward), where:
            next_state: the new state after the actions.
            reward: a dictionary mapping each agent to its reward.
        """
        pass




# -----------------------
# Legacy Environment
# -----------------------

class RLEnvironment(Environment):
    def __init__(self, transition_rule, reward_rule, state):
        """
        RLEnvironment that manages multi-agent setups, data generation, and reward computation.
        """
        super().__init__(transition_rule=transition_rule, reward_rule=reward_rule, state=state)

    def reset(self, transition_rule=None, reward_rule=None, state=None):
        """
        Resets the environment state
        """
        super.reset(transition_rule, reward_rule, state)


    def step(self, actions: Dict[Any, Any]) -> Tuple[Any, Dict[Any, float]]:
        """
        Generates new data for this step, updates the state, and returns the updated state along with computed rewards.
        Actions are ignored in the legacy environment's default transition.
        
        :param actions: (Ignored) Dictionary mapping agent IDs to their actions.
        :return: A tuple (new_state, rewards), where rewards is a dict mapping agent IDs to their computed rewards.
        """
        # Store the current state for reward computation.
        previous_state = self.state
        
        # Delegate dataset generation and state update to the transition rule.
        new_state = self.transition_rule(previous_state, actions)

        # Compute rewards using the reward_rule provided
        rewards = self.reward_rule(previous_state, actions, new_state)
        
        # Update the environment's state.
        self.state = new_state
        
        return new_state, rewards


