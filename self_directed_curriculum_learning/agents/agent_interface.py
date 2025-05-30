# agents/agent_interface.py

from abc import ABC, abstractmethod

class AgentInterface(ABC):
    def __init__(self, training_params=None):
        self.actions_history = []
        self.valid_actions = []
        self.training_params = training_params


    @abstractmethod
    def choose_action(self, state, step):
        """
        Choose an action based on the current state and internal logic.
        """
        pass

    @abstractmethod
    def evaluate_accuracy(self, model, dataset):
        """
        Evaluate the model's accuracy on the provided dataset.
        """
        pass

    def update_valid_actions(self, new_valid_actions):
        """
        Update the list of valid actions that the agent can select from.
        
        Args:
            new_valid_actions (list): The new list of valid actions.
        """
        self.valid_actions = new_valid_actions.copy()

    def get_actions_history(self):
        """
        Returns the history of the actions taken by the agent.
        
        Returns:
            list: A list of actions taken by the agent.
        """
        return self.actions_history
