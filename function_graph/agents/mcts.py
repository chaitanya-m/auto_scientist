# agents/mcts.py
from agents.mcts_agent_interface import MCTSAgentInterface
from data_gen.curriculum import Curriculum
from utils.nn import create_minimal_graphmodel
from graph.composer import GraphComposer, GraphTransformer
from graph.node import SingleNeuron, SubGraphNode
import uuid
import os
import math
import random
import tensorflow as tf
from keras import models, layers
import numpy as np

def compute_complexity(composer):
    """
    Compute the complexity of a given graph.
    For now, we use a simple measure: the number of nodes in the composer.
    """
    return len(composer.nodes)

class PolicyNetwork:
    """
    A simple MLP policy network which maps state feature vectors to a probability 
    distribution over actions.
    
    For this example, the network takes a fixed-size input (3 features) and outputs
    a fixed-length probability vector (3 actions), corresponding to:
      ["add_neuron", "delete_repository_entry", "add_from_repository"]
    """
    def __init__(self, input_dim=3, num_actions=3, hidden_units=16, learning_rate=0.001):
        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(num_actions, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy')

    def predict(self, features):
        """
        Predicts a probability distribution over the full action set.

        Args:
            features (np.array): Input feature vector with shape (input_dim,) or (batch, input_dim).

        Returns:
            np.array: Probability vector of shape (num_actions,).
        """
        # Ensure batch dimension.
        features = np.atleast_2d(features)
        return self.model.predict(features, verbose=0)[0]

    def train(self, features, actions, rewards, epochs=5, batch_size=32):
        """
        Trains the policy network on a batch of experiences.

        Args:
            features (np.array): Feature vectors with shape (batch, input_dim).
            actions (np.array): Integer indices for actions with shape (batch,).
            rewards (np.array): Reward values used as sample weights.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Train using sample weights to emphasize more rewarded samples.
        self.model.fit(features, actions, sample_weight=rewards, 
                       epochs=epochs, batch_size=batch_size, verbose=0)

class SimpleMCTSAgent(MCTSAgentInterface):
    class TreeNode:
        """
        Represents a node in the MCTS tree.
        Stores the state, parent reference, the action taken to reach this state,
        children (mapping action -> TreeNode), visit counts, and total reward.
        """
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action  # Action taken to get here from parent.
            self.children = {}    # Mapping from action to TreeNode.
            self.visits = 0
            self.total_value = 0.0  # Sum of rewards.

        def average_value(self):
            """Return the average reward from this node."""
            return self.total_value / self.visits if self.visits > 0 else 0.0

    def __init__(self):
        # Initialize the curriculum and reference autoencoder.
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.input_dim = self.reference['config']['input_dim']
        self.latent_dim = self.reference['config']['encoder'][-1]
        
        # Extract the reference encoder from the full, trained autoencoder.
        self.reference_encoder = self.get_reference_encoder()
        
        # Global repository to store subgraphs that improved performance.
        # Each entry is a dict with: subgraph, performance, action history, and utility (-performance).
        self.repository = []
        self.best_mse = float('inf')
        
        # Global counters for monitoring deletions and improvements.
        self.deletion_count = 0
        self.improvement_count = 0
        
        # Root of the search tree.
        self.root = None

        # Initialize the policy network and experience buffer.
        # Fixed feature vector: [current candidate MSE, # of actions, repository size].
        self.policy_net = PolicyNetwork(input_dim=3, num_actions=3)
        self.experience = []  # Experience tuples: (features, action index, reward)
        self.experience_threshold = 20  # Trigger training after collecting this many samples.

    def get_reference_encoder(self):
        """
        Extracts the encoder component from the full reference autoencoder.
        Assumes the autoencoder is a Sequential model and that the encoder corresponds to the 
        first len(config["encoder"]) layers.
        """
        config = self.reference["config"]
        ref_model = self.reference["autoencoder"]
        encoder = models.Sequential()
        encoder.add(layers.InputLayer(input_shape=(config["input_dim"],)))
        for i in range(len(config["encoder"])):
            encoder.add(ref_model.layers[i])
        return encoder

    def get_initial_state(self):
        """
        Creates the initial state for the MCTS tree.
        Returns:
            dict: State with a new GraphComposer, empty action history, dummy performance, and target MSE.
        """
        composer, _ = create_minimal_graphmodel(
            (self.input_dim,),
            output_units=self.latent_dim,
            activation="relu"
        )
        return {
            "composer": composer,
            "graph_actions": [],
            "performance": 1.0,  # Dummy initial performance.
            "target_mse": self.target_mse,
        }
    
    def get_available_actions(self, state):
        """
        Returns the list of valid actions in the current state.
        """
        actions = ["add_neuron", "delete_repository_entry"]
        if self.repository:
            actions.append("add_from_repository")
        return actions

    def apply_action(self, state, action):
        """
        Applies the specified action to the state, updating the GraphComposer accordingly.

        For "add_neuron", adds a new neuron node.
        For "delete_repository_entry", randomly removes an entry from the repository.
        For "add_from_repository", reuses the repository entry with the highest utility.

        Returns:
            dict: New state after the action is applied.
        """
        composer = state["composer"]
        new_state = state.copy()
        new_state["graph_actions"] = state["graph_actions"] + [action]
        
        if action == "add_neuron":
            new_node = SingleNeuron(name=str(uuid.uuid4()), activation="relu")
            try:
                composer.disconnect("input", "output")
            except Exception:
                pass
            composer.add_node(new_node)
            composer.connect("input", new_node.name)
            composer.connect(new_node.name, "output")
        
        elif action == "delete_repository_entry":
            if self.repository:
                idx = random.randrange(len(self.repository))
                del self.repository[idx]
                self.deletion_count += 1  # Update deletion counter.
                
        elif action == "add_from_repository":
            if self.repository:
                # Select the repository entry with the highest utility.
                best_entry = max(self.repository, key=lambda entry: entry["utility"])
                transformer = GraphTransformer(composer)
                transformer.add_abstraction_node(
                    best_entry["subgraph_node"],
                    chosen_subset=["input"],
                    outputs=["output"]
                )
        
        composer.keras_model = None
        composer.build()
        new_state["performance"] = state["performance"]
        return new_state

    def get_training_data(self):
        """
        Returns dummy training data for candidate evaluation.
        """
        X = np.random.rand(100, self.input_dim)
        return X, None

    def update_repository(self, state):
        """
        Checks if the candidate encoder's performance is improved relative to the best MSE.
        If so, updates the repository with a new subgraph and increments improvement count.
        Computes utility as negative performance.
        """
        if state["performance"] < self.best_mse:
            self.best_mse = state["performance"]
            self.improvement_count += 1
            tmp_filepath = f"temp_subgraph_{uuid.uuid4().hex[:4]}.keras"
            state["composer"].save_subgraph(tmp_filepath)
            new_subgraph_node = SubGraphNode.load(tmp_filepath, name=f"subgraph_{uuid.uuid4().hex[:4]}")
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            transformer = GraphTransformer(state["composer"])
            transformer.add_abstraction_node(
                new_subgraph_node,
                chosen_subset=["input"],
                outputs=["output"]
            )
            # Compute utility: lower performance (MSE) gives higher utility.
            repo_entry = {
                "subgraph_node": new_subgraph_node,
                "performance": state["performance"],
                "graph_actions": state["graph_actions"].copy(),
                "utility": -state["performance"]
            }
            self.repository.append(repo_entry)
            print(f"Repository updated: New best performance {self.best_mse}")

    def evaluate_state(self, state):
        """
        Evaluates the candidate encoder represented by the state.
        Trains the candidate briefly and computes its validation MSE.
        Also computes a reward based on performance and complexity.
        
        Reward formula:
            Reward = (Reference Performance / Reference Complexity) - (Candidate MSE / Candidate Complexity)
        
        Complexity is measured simply as the number of nodes in the GraphComposer.
        
        Returns:
            float: The candidate encoder's MSE.
        """
        # Obtain training data.
        X, _ = self.get_training_data()
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        # Get target latent representations from the reference encoder.
        y_train = self.reference_encoder.predict(X_train)
        y_test = self.reference_encoder.predict(X_test)
        
        # Build the candidate encoder model.
        model = state["composer"].build()
        model.compile(optimizer="adam", loss="mse")
        
        # Train candidate encoder.
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=0)
        mse = history.history["val_loss"][-1]
        state["performance"] = mse
        
        # Compute complexity metrics.
        candidate_complexity = compute_complexity(state["composer"])
        # For reference, we use a simple proxy: number of layers in the encoder.
        reference_complexity = len(self.reference["config"]["encoder"])
        
        reference_score = self.target_mse / reference_complexity
        candidate_score = mse / candidate_complexity
        reward = reference_score - candidate_score
        
        self.update_repository(state)
        
        return mse  # Reward is used in record_experience.
    
    def extract_features(self, state):
        """
        Constructs a feature vector from the current state.
        
        The features used are:
          - Current candidate MSE (lower is better)
          - Number of actions taken in the state's history
          - Current repository size
        
        Returns:
            np.array: Feature vector [performance, number of actions, repository size].
        """
        perf = state.get("performance", 1.0)
        num_actions = len(state.get("graph_actions", []))
        repo_size = len(self.repository)
        return np.array([perf, num_actions, repo_size], dtype=float)

    def policy_network(self, state, actions):
        """
        Uses the learned policy network to predict probabilities over available actions.
        
        The policy network always outputs a fixed-length probability vector corresponding
        to all possible actions in the full action set:
            ["add_neuron", "delete_repository_entry", "add_from_repository"]
        
        Since not all actions may be available in a given state (e.g., if the repository is empty),
        we do the following:
          1. Obtain the full probability vector (e.g., [p1, p2, p3]).
          2. Create an availability mask (1 for available actions, 0 for unavailable).
          3. Multiply element-wise to zero out unavailable actions.
          4. Renormalize the resulting vector so that the probabilities sum to 1.
        
        Returns:
            List[float]: Probabilities corresponding only to the available actions.
        """
        full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
        features = self.extract_features(state)
        probs_full = self.policy_net.predict(features)  # shape (3,)
        
        # Action masking and filtering.
        available_probs = []
        for action in actions:
            idx = full_actions.index(action)
            available_probs.append(probs_full[idx])
        available_probs = np.array(available_probs)
        if available_probs.sum() == 0:
            available_probs = np.ones(len(actions))
        available_probs = available_probs / available_probs.sum()
        return available_probs.tolist()

    def record_experience(self, state, action, reward):
        """
        Records an experience tuple (features, action index, reward) and triggers training
        of the policy network if the experience buffer reaches a specified threshold.
        
        Args:
            state (dict): The current state.
            action (str): The action taken.
            reward (float): The reward observed.
        """
        features = self.extract_features(state)
        full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
        try:
            action_idx = full_actions.index(action)
        except ValueError:
            return
        self.experience.append((features, action_idx, reward))
        if len(self.experience) >= self.experience_threshold:
            self.train_policy_network()
            self.experience = []  # Clear experience buffer after training.

    def train_policy_network(self):
        """
        Trains the policy network on all recorded experiences.
        
        Uses the reward as a sample weight to give higher importance to better experiences.
        """
        features = np.array([exp[0] for exp in self.experience])
        actions = np.array([exp[1] for exp in self.experience])
        rewards = np.array([exp[2] for exp in self.experience])
        self.policy_net.train(features, actions, rewards)
        print("Policy network trained on experience.")

    def is_terminal(self, state):
        """
        Determines if the state is terminal.
        For now, our agent does not have a terminal state condition.
        """
        return False

    def mcts_search(self, search_budget=20, exploration_constant=1.41):
        """
        Performs Monte Carlo Tree Search (MCTS) for a given search budget.
        
        - Initializes the root state.
        - Performs a selection/expansion/evaluation loop for 'search_budget' iterations.
        - Records experiences using the computed reward.
        - Updates node statistics (visits and total value) for backpropagation.
        
        At the end of the search, prints overall statistics including the final repository size,
        total deletion actions, and improvement events.
        
        Returns:
            dict: The state corresponding to the best node found.
        """
        root_state = self.get_initial_state()
        self.root = self.TreeNode(root_state)
        
        for iteration in range(search_budget):
            node = self.root
            # Selection phase.
            while node.children:
                available_actions = self.get_available_actions(node.state)
                probs = self.policy_network(node.state, available_actions)
                best_ucb = -float('inf')
                best_action = None
                best_child = None
                full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
                for i, action in enumerate(full_actions):
                    if action not in available_actions:
                        continue  # Skip actions not available.
                    if action in node.children:
                        child = node.children[action]
                        avg_reward = child.average_value()
                        ucb = avg_reward + exploration_constant * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
                        ucb *= probs[available_actions.index(action)]
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_action = action
                            best_child = child
                    else:
                        ucb = exploration_constant * math.sqrt(math.log(node.visits + 1))
                        ucb *= probs[available_actions.index(action)]
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_action = action
                            best_child = None
                if best_child is None:
                    new_state = self.apply_action(node.state, best_action)
                    new_node = self.TreeNode(new_state, parent=node, action=best_action)
                    node.children[best_action] = new_node
                    # Evaluate the new state and record the experience (using a default reward of 0 here).
                    # We will later use evaluate_state to update the reward.
                    _ = self.evaluate_state(new_state)
                    self.record_experience(node.state, best_action, 0)
                    node = new_node
                    break
                else:
                    node = best_child
            
            # Evaluation phase for the leaf node.
            reward = -self.evaluate_state(node.state)
            # Record experience for the parent node of the leaf.
            if node.parent is not None and node.action is not None:
                self.record_experience(node.parent.state, node.action, reward)
            # Backpropagation: update all nodes on the path from the leaf to the root.
            temp = node
            while temp is not None:
                temp.visits += 1
                temp.total_value += reward
                temp = temp.parent

        print("MCTS search completed.")
        print(f"Final repository size: {len(self.repository)}")
        print(f"Total deletion actions: {self.deletion_count}")
        print(f"Total improvement events: {self.improvement_count}")

        # Identify the best node by scanning all nodes.
        best_node = self.root
        best_avg = best_node.average_value()
        nodes_to_check = [self.root]
        while nodes_to_check:
            current = nodes_to_check.pop()
            if current.average_value() > best_avg:
                best_avg = current.average_value()
                best_node = current
            nodes_to_check.extend(list(current.children.values()))
        return best_node.state
