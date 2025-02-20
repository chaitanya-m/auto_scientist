# agents/deterministic.py

from agents.agent_interface import AgentInterface
import numpy as np

# -----------------------
# Deterministic Agent
# -----------------------
class DeterministicAgent(AgentInterface):
    def __init__(self, policy=None):
        """
        Initializes an agent with a predetermined sequence of actions.
        The action plan is a list of actions to be performed in sequence.
        If the action plan is exhausted, the agent selects actions randomly.
        """
        super().__init__()
        # Use the provided policy, or fall back to a default one that picks randomly.
        self.policy = policy if policy is not None else self.default_policy

    def default_policy(self, state, step, valid_actions):
        """
        A simple policy that chooses randomly among whatever valid_actions are provided.
        """
        return np.random.choice(valid_actions) if valid_actions else None

    def set_policy(self, new_policy):
        """
        Setter to update the policy callable at runtime.
        """
        self.policy = new_policy

    def choose_action(self, state, step):
        """
        Invoke the policy function to select an action from self.valid_actions.
        """
        action = self.policy(state, step, self.valid_actions)
        self.actions_history.append(action)
        return action

    def evaluate_accuracy(self, model, dataset):
        """
        Uses a shared utility function to train and evaluate the model
        with a parameterized train/test split.
        """
        from utils.nn import train_and_evaluate
        
        # Default ratio or retrieve from training_params if desired.
        ratio = self.training_params.get("train_ratio", 0.5)

        return train_and_evaluate(
            model=model,
            dataset=dataset,
            train_ratio=ratio,
            epochs=self.training_params["epochs"],
            verbose=self.training_params["verbose"]
        )


    def get_actions_history(self):
        """
        Returns the history of the actions taken by the agent.
        
        Returns:
            list: A list of actions taken by the agent.
        """
        return self.actions_history