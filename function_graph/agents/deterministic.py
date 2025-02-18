# agents/deterministic.py

import numpy as np

# -----------------------
# Deterministic Agent
# -----------------------
class DeterministicAgent:
    def __init__(self, action_plan=None):
        """
        Initializes an agent with a predetermined sequence of actions.
        The action plan is a list of actions to be performed in sequence.
        If the action plan is exhausted, the agent selects actions randomly.
        """
        self.actions_history = []
        self.action_plan = action_plan

    def choose_action(self, state, valid_actions):
        """
        Chooses an action based on the predetermined action plan if available;
        otherwise, picks a random action from valid_actions.

        Args:
            state: The current state of the environment (unused in this deterministic agent).
            valid_actions (list): List of valid actions.

        Returns:
            The chosen action.
        """
        if self.action_plan:
            action = self.action_plan.pop(0)
        else:
            action = np.random.choice(valid_actions)
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
        
        model.fit(train_features, train_labels, epochs=1, verbose=0)
        predictions = model.predict(test_features, verbose=0)
        preds = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(preds == test_labels)
        return accuracy
