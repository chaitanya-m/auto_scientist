import numpy as np

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99):
        """
        action_space: List of possible actions (e.g., ["add_abstraction", "no_change"]).
        alpha: learning rate
        gamma: discount factor
        epsilon: initial exploration rate
        epsilon_decay: factor by which epsilon is multiplied each step/episode
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # e.g., {(some_state_representation, action): q_value}
        
        # For storing (state, action) to update after we see the reward
        self.last_state = None
        self.last_action = None

        # For storing actions history if needed
        self.actions_history = {}

    def choose_action(self, agent_id, state, valid_actions):
        """
        1. Convert 'state' (or relevant parts) to a discrete representation or key.
        2. With probability self.epsilon, pick a random action from valid_actions.
           Otherwise pick the best action based on self.q_table.
        3. Store (state, action) in self.last_state/last_action for later Q-update.
        """
        # Example pseudo-logic. Adjust as needed for your environment’s state info.
        state_key = self._make_state_key(state, agent_id)
        
        if np.random.random() < self.epsilon:
            # Explore
            action = np.random.choice(valid_actions)
        else:
            # Exploit
            q_values_for_state = [self.q_table.get((state_key, a), 0.0) for a in valid_actions]
            best_idx = int(np.argmax(q_values_for_state))
            action = valid_actions[best_idx]
        
        # Store for Q-update
        self.last_state = state_key
        self.last_action = action
        
        # Decrease epsilon over time
        self.epsilon *= self.epsilon_decay
        
        # Also store for logging if needed
        if agent_id not in self.actions_history:
            self.actions_history[agent_id] = []
        self.actions_history[agent_id].append(action)
        
        return action
    
    def update_q(self, reward, next_state, agent_id, done=False):
        """
        Called after we receive a reward and observe next_state.
        next_state can be used to do the Q-learning update:
            Q(s, a) <- Q(s, a) + alpha * [ r + gamma*max_a' Q(s', a') - Q(s, a) ]
        """
        if self.last_state is None or self.last_action is None:
            return

        state_key = self.last_state
        action = self.last_action
        next_state_key = self._make_state_key(next_state, agent_id)
        
        old_q = self.q_table.get((state_key, action), 0.0)
        
        # If we are done, future returns are zero, else we pick best next action from Q
        if done:
            future = 0.0
        else:
            # check all possible actions from the next state
            future_qs = [self.q_table.get((next_state_key, a), 0.0) for a in self.action_space]
            future = self.gamma * max(future_qs) if future_qs else 0.0

        new_q = old_q + self.alpha * (reward + future - old_q)
        self.q_table[(state_key, action)] = new_q

        # We could reset last_state/last_action, or keep them until next step
        if done:
            self.last_state = None
            self.last_action = None

    def evaluate_accuracy(self, model, dataset):
        """
        Similar to DummyAgent, we do a 50/50 train/test split, train the model on train data,
        evaluate accuracy on test data, then return the accuracy. 
        This can remain the same or be specialized for Q-learning if needed.
        """
        split_idx = len(dataset) // 2
        train_df = dataset.iloc[:split_idx]
        test_df = dataset.iloc[split_idx:]
        
        train_features = train_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        train_labels = train_df["label"].to_numpy(dtype=int)
        
        test_features = test_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        test_labels = test_df["label"].to_numpy(dtype=int)
        
        # Train the model
        model.fit(train_features, train_labels, epochs=1, verbose=0)
        
        # Evaluate on test
        predictions = model.predict(test_features, verbose=0)
        preds = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(preds == test_labels)
        return accuracy

    def _make_state_key(self, state, agent_id):
        """
        Converts the environment state into a simplified representation to be used
        as a key for the Q-table. In this example, we use a tuple consisting of:
          - The current step number.
          - A boolean flag indicating whether any learned abstraction node has been added.
            (Since abstraction nodes receive unique suffixes, we check if any node name 
             starts with "learned_abstraction".)
        """
        # Get the current step number from the state dictionary (default to 0 if not present).
        step_num = state.get("step", 0)
        
        # Extract the list of node names for the given agent.
        agent_nodes = state["agents_networks"][agent_id]["nodes"]
        
        # Determine if an abstraction node has been added by checking if any node name
        # starts with "learned_abstraction". This works even if a unique suffix was appended.
        has_abstraction = any(node_name.startswith("learned_abstraction") for node_name in agent_nodes)
        
        # Return the state key as a tuple: (step number, presence of an abstraction node)
        return (step_num, has_abstraction)
