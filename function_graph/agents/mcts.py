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

class PolicyNetwork:
    """
    A simple MLP policy network which maps state feature vectors to probability distributions over actions.
    For this example, we define a fixed input dimension (3 features) and fixed output dimension (3 actions).
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
        features: a numpy array of shape (input_dim, ) or (batch, input_dim)
        returns: a numpy array of probabilities (if batch, then shape (batch, num_actions)).
        """
        # Ensure batch dimension.
        features = np.atleast_2d(features)
        return self.model.predict(features, verbose=0)[0]

    def train(self, features, actions, rewards, epochs=5, batch_size=32):
        """
        Train the policy network on a batch of experiences.
        features: numpy array of shape (batch, input_dim)
        actions: numpy array of shape (batch,) as integer indices (0,1,2)
        rewards: numpy array of shape (batch,) used as sample weights.
        """
        # Train using sample weights to weight the loss by observed reward.
        self.model.fit(features, actions, sample_weight=rewards, 
                       epochs=epochs, batch_size=batch_size, verbose=0)

class SimpleMCTSAgent(MCTSAgentInterface):
    class TreeNode:
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action  # Action taken to get here from parent.
            self.children = {}    # Mapping from action to TreeNode.
            self.visits = 0
            self.total_value = 0.0  # Sum of rewards (reward = -performance).

        def average_value(self):
            return self.total_value / self.visits if self.visits > 0 else 0.0

    def __init__(self):
        # Initialize the curriculum and retrieve the precomputed reference autoencoder.
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.input_dim = self.reference['config']['input_dim']
        self.latent_dim = self.reference['config']['encoder'][-1]
        
        # Extract the reference encoder from the full, trained reference autoencoder.
        self.reference_encoder = self.get_reference_encoder()
        
        # Repository to store successful architectures.
        self.repository = []  # Each entry will have an associated 'utility' computed as -performance.
        self.best_mse = float('inf')
        
        # Root of the search tree.
        self.root = None

        # Initialize the policy network and experience buffer.
        # In this example we use a fixed feature vector of size 3 and 3 output actions.
        self.policy_net = PolicyNetwork(input_dim=3, num_actions=3)
        self.experience = []  # Each experience is a tuple (features, action_index, reward)
        self.experience_threshold = 20  # Train when at least this many samples are collected

    def get_reference_encoder(self):
        """
        Extracts the encoder component from the full reference autoencoder.
        Assumes the reference autoencoder is a Sequential model and the encoder
        corresponds to the first len(config["encoder"]) layers.
        """
        config = self.reference["config"]
        ref_model = self.reference["autoencoder"]  # The full, trained autoencoder.
        encoder = models.Sequential()
        encoder.add(layers.InputLayer(input_shape=(config["input_dim"],)))
        for i in range(len(config["encoder"])):
            encoder.add(ref_model.layers[i])
        return encoder

    def get_initial_state(self):
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
        actions = ["add_neuron", "delete_repository_entry"]
        if self.repository:
            actions.append("add_from_repository")
        return actions

    def apply_action(self, state, action):
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
                # Remove a random repository entry.
                idx = random.randrange(len(self.repository))
                del self.repository[idx]
                
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
        import numpy as np
        X = np.random.rand(100, self.input_dim)
        return X, None

    def update_repository(self, state):
        if state["performance"] < self.best_mse:
            self.best_mse = state["performance"]
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
            # Compute utility as -performance (lower performance/MSE implies higher utility).
            repo_entry = {
                "subgraph_node": new_subgraph_node,
                "performance": state["performance"],
                "graph_actions": state["graph_actions"].copy(),
                "utility": -state["performance"]
            }
            self.repository.append(repo_entry)
            print(f"Repository updated: New best performance {self.best_mse}")

    def evaluate_state(self, state):
        # Obtain a batch of training data.
        X, _ = self.get_training_data()
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        # Use the reference encoder to generate target latent representations.
        y_train = self.reference_encoder.predict(X_train)
        y_test = self.reference_encoder.predict(X_test)
        
        # Build and compile the candidate encoder from the current state.
        model = state["composer"].build()
        model.compile(optimizer="adam", loss="mse")
        
        # Train the candidate encoder briefly.
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=0)
        mse = history.history["val_loss"][-1]
        state["performance"] = mse
        
        # Update repository if performance improved.
        self.update_repository(state)
        
        return mse

    def extract_features(self, state):
        """
        Constructs a feature vector from the current state.
        Here we use:
          - performance: current candidate MSE (lower is better)
          - number of actions taken
          - size of the repository
        """
        perf = state.get("performance", 1.0)
        num_actions = len(state.get("graph_actions", []))
        repo_size = len(self.repository)
        # You may choose to scale or transform these features.
        return np.array([perf, num_actions, repo_size], dtype=float)

    def policy_network(self, state, actions):
        """
        Uses the learned policy network to predict probabilities over available actions.
        
        The policy network always outputs a fixed-length probability vector corresponding
        to all possible actions in the full action set:
            [ "add_neuron", "delete_repository_entry", "add_from_repository" ]
        
        Since not all these actions may be available in a given state (for instance, if the
        repository is empty "add_from_repository" might be invalid), we perform the following steps:
        
        1. Full Probability Vector: The network outputs a vector, e.g., [p1, p2, p3].
        2. Action Masking: Create a mask indicating available actions (1 for allowed, 0 for disallowed).
           For example, if available actions are ["add_neuron", "delete_repository_entry"], the mask is [1, 1, 0].
        3. Filtering: Multiply the full probability vector element-wise by the mask. This sets the
           probabilities of unavailable actions to zero.
        4. Renormalization: Divide each element of the filtered vector by the total sum of the vector,
           ensuring that the probabilities for the available actions sum to 1.
        
        Returns:
            A list of probabilities corresponding only to the available actions.
        """
        # Full action set for reference.
        full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
        # Extract state features.
        features = self.extract_features(state)
        probs_full = self.policy_net.predict(features)  # shape (3,)
        
        # --- Action Masking and Filtering ---
        # Filter probabilities to only those corresponding to available actions.
        available_probs = []
        for action in actions:
            idx = full_actions.index(action)
            available_probs.append(probs_full[idx])
        # Renormalize the available probabilities.
        available_probs = np.array(available_probs)
        if available_probs.sum() == 0:
            available_probs = np.ones(len(actions))
        available_probs = available_probs / available_probs.sum()
        return available_probs.tolist()


    def record_experience(self, state, action, reward):
        """
        Records an experience tuple and triggers training if enough data has been collected.
        'action' is expected to be one of the strings in the full action space.
        """
        features = self.extract_features(state)
        full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
        try:
            action_idx = full_actions.index(action)
        except ValueError:
            return  # Unrecognized action.
        self.experience.append((features, action_idx, reward))
        # If we have enough experience, train the policy network.
        if len(self.experience) >= self.experience_threshold:
            self.train_policy_network()
            self.experience = []  # Clear after training.

    def train_policy_network(self):
        """
        Trains the policy network on all recorded experiences.
        We use the observed reward as a weight for each sample.
        """
        features = np.array([exp[0] for exp in self.experience])
        actions = np.array([exp[1] for exp in self.experience])
        rewards = np.array([exp[2] for exp in self.experience])
        self.policy_net.train(features, actions, rewards)
        print("Policy network trained on experience.")

    def is_terminal(self, state):
        return False

    def mcts_search(self, search_budget=20, exploration_constant=1.41):
        root_state = self.get_initial_state()
        self.root = self.TreeNode(root_state)
        
        for iteration in range(search_budget):
            node = self.root
            # Select phase.
            while node.children:
                available_actions = self.get_available_actions(node.state)
                probs = self.policy_network(node.state, available_actions)
                best_ucb = -float('inf')
                best_action = None
                best_child = None
                full_actions = ["add_neuron", "delete_repository_entry", "add_from_repository"]
                for i, action in enumerate(full_actions):
                    if action not in available_actions:
                        continue  # Skip actions not allowed in current state.
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
                    # Record the experience immediately after node expansion.
                    # We use a negative reward (since performance is MSE, lower is better).
                    self.record_experience(node.state, best_action, 0)
                    node = new_node
                    break
                else:
                    node = best_child
            
            # Expand/evaluate leaf.
            reward = -self.evaluate_state(node.state)
            # Record the final experience from this simulation.
            if node.parent is not None and node.action is not None:
                self.record_experience(node.parent.state, node.action, reward)
            temp = node
            while temp is not None:
                temp.visits += 1
                temp.total_value += reward
                temp = temp.parent
        
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


