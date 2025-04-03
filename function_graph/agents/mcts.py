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
        # Initialize the curriculum and reference autoencoder parameters.
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.input_dim = self.reference['config']['input_dim']
        self.latent_dim = self.reference['config']['encoder'][-1]
        
        # Repository to store successful architectures.
        self.repository = []
        self.best_mse = float('inf')
        
        # Root of the search tree.
        self.root = None

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
                repo_entry = self.repository[0]  # For now, simply use the first entry.
                transformer = GraphTransformer(composer)
                transformer.add_abstraction_node(
                    repo_entry["subgraph_node"],
                    chosen_subset=["input"],
                    outputs=["output"]
                )
                
        composer.keras_model = None
        composer.build()
        new_state["performance"] = state["performance"]
        return new_state

    def evaluate_state(self, state):
        X, y = self.get_training_data()
        model = state["composer"].build()
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=5, verbose=0)
        mse = history.history['loss'][-1]
        state["performance"] = mse
        self.update_repository(state)
        return mse

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
            repo_entry = {
                "subgraph_node": new_subgraph_node,
                "performance": state["performance"],
                "graph_actions": state["graph_actions"].copy()
            }
            self.repository.append(repo_entry)
            print(f"Repository updated: New best performance {self.best_mse}")

    def get_training_data(self):
        import numpy as np
        X = np.random.rand(100, self.input_dim)
        # For encoder evaluation, use the first latent_dim columns of X as target.
        y = X[:, :self.latent_dim]
        return X, y

    def policy_network(self, state, actions):
        # Stub: return uniform probabilities over available actions.
        n = len(actions)
        return [1.0 / n for _ in actions]

    def is_terminal(self, state):
        return False

    def mcts_search(self, search_budget=20, exploration_constant=1.41):
        # Initialize the tree with the root node.
        root_state = self.get_initial_state()
        self.root = self.TreeNode(root_state)
        
        for iteration in range(search_budget):
            # Selection: start from root and traverse until a leaf is reached.
            node = self.root
            while node.children:
                actions = self.get_available_actions(node.state)
                probs = self.policy_network(node.state, actions)
                best_ucb = -float('inf')
                best_action = None
                best_child = None
                for i, action in enumerate(actions):
                    if action in node.children:
                        child = node.children[action]
                        avg_reward = child.average_value()
                        ucb = avg_reward + exploration_constant * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
                        ucb *= probs[i]
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_action = action
                            best_child = child
                    else:
                        ucb = exploration_constant * math.sqrt(math.log(node.visits + 1))
                        ucb *= probs[i]
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_action = action
                            best_child = None
                if best_child is None:
                    new_state = self.apply_action(node.state, best_action)
                    new_node = self.TreeNode(new_state, parent=node, action=best_action)
                    node.children[best_action] = new_node
                    node = new_node
                    break
                else:
                    node = best_child
            
            # Simulation: evaluate the selected node.
            reward = -self.evaluate_state(node.state)  # Reward is negative MSE.
            # Backpropagation: update nodes along the path.
            temp = node
            while temp is not None:
                temp.visits += 1
                temp.total_value += reward
                temp = temp.parent
        
        # After search, select the best node based on average value.
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
