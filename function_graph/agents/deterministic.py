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
        import numpy as np
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
        Performs a 50/50 train/test split on the dataset, trains the model on the training
        split, and returns the test accuracy.

        Args:
            model: The Keras model to be evaluated.
            dataset (DataFrame): The dataset containing features and labels.

        Returns:
            The accuracy computed on the test split.
        """
        split_idx = len(dataset) // 2
        train_df = dataset.iloc[:split_idx]
        test_df = dataset.iloc[split_idx:]
        
        train_features = train_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        train_labels = train_df["label"].to_numpy(dtype=int)
        
        test_features = test_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        test_labels = test_df["label"].to_numpy(dtype=int)
        
        model.fit(train_features, 
                  train_labels, 
                  epochs=self.training_params["epochs"], 
                  verbose=self.training_params["verbose"]
                  )
        
        predictions = model.predict(test_features, verbose=0)
        preds = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(preds == test_labels)
        return accuracy

    def get_actions_history(self):
        """
        Returns the history of the actions taken by the agent.
        
        Returns:
            list: A list of actions taken by the agent.
        """
        return self.actions_history