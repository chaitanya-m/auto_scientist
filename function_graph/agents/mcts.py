# agents/mcts.py
from agents.mcts_agent_interface import MCTSAgentInterface
from data_gen.curriculum import Curriculum
from utils.nn import create_minimal_graphmodel
from graph.composer import GraphComposer, GraphTransformer
from graph.node import SingleNeuron, SubGraphNode
import uuid
import os

class SimpleMCTSAgent(MCTSAgentInterface):
    def __init__(self):
        # Initialize the curriculum and get reference autoencoder parameters.
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.input_dim = self.reference['config']['input_dim']
        self.latent_dim = self.reference['config']['encoder'][-1]
        
        # Repository to store successful architectures (starting empty).
        self.repository = []
        self.best_mse = float('inf')

    def get_initial_state(self):
        # Build a minimal graph with an input shape matching the reference
        # and an output corresponding to the encoder's latent space.
        composer, _ = create_minimal_graphmodel(
            (self.input_dim,),
            output_units=self.latent_dim,
            activation="relu"
        )
        return {
            "composer": composer,
            "graph_actions": [],
            "performance": 1.0,  # Dummy performance for now
            "target_mse": self.target_mse,
        }
    
    def get_available_actions(self, state):
        # For now, we have three actions: add a neuron, add from repository, or delete a repository entry.
        actions = ["add_neuron", "delete_repository_entry"]
        if self.repository:
            actions.append("add_from_repository")
        return actions
    
    def apply_action(self, state, action):
        composer = state["composer"]
        new_state = state.copy()
        new_state["graph_actions"] = state["graph_actions"] + [action]
        
        if action == "add_neuron":
            # Add a new neuron (using SingleNeuron for now) to the graph.
            new_node = SingleNeuron(name=str(uuid.uuid4()), activation="relu")
            # For simplicity, assume we add the neuron between input and output.
            try:
                composer.disconnect("input", "output")
            except Exception:
                pass  # If the connection doesn't exist, skip
            composer.add_node(new_node)
            composer.connect("input", new_node.name)
            composer.connect(new_node.name, "output")
        
        elif action == "delete_repository_entry":
            # Remove the last entry from the repository.
            if self.repository:
                self.repository.pop()
                
        elif action == "add_from_repository":
            if self.repository:
                # Retrieve the first repository entry.
                repo_entry = self.repository[0]
                # Use GraphTransformer to add the repository subgraph into the current graph.
                transformer = GraphTransformer(composer)
                # For simplicity, we choose "input" as the chosen subset and "output" as the output.
                transformer.add_abstraction_node(
                    repo_entry["subgraph_node"],
                    chosen_subset=["input"],
                    outputs=["output"]
                )
        
        # Clear any built model so that a new one is constructed on next evaluation.
        composer.keras_model = None
        # Rebuild the model.
        composer.build()
        
        # Dummy update: performance remains unchanged until evaluate_state is called.
        new_state["performance"] = state["performance"]
        return new_state

    def evaluate_state(self, state):
        """
        Evaluate the state's performance by training the model briefly and computing MSE.
        """
        X, y = self.get_training_data()
        model = state["composer"].build()
        model.compile(optimizer="adam", loss="mse")
        
        # Train briefly for fast evaluation.
        history = model.fit(X, y, epochs=5, verbose=0)
        mse = history.history['loss'][-1]
        state["performance"] = mse
        
        # Update repository if this state is better than the current best.
        self.update_repository(state)
        return mse

    def update_repository(self, state):
        """
        Add state to the repository if its performance improves on the current best.
        Uses save_subgraph and add_abstraction_node to store the subgraph.
        """
        if state["performance"] < self.best_mse:
            self.best_mse = state["performance"]
            # Save the current subgraph to a temporary file.
            tmp_filepath = f"temp_subgraph_{uuid.uuid4().hex[:4]}.keras"
            state["composer"].save_subgraph(tmp_filepath)
            # Load the saved subgraph as a SubGraphNode.
            new_subgraph_node = SubGraphNode.load(tmp_filepath, name=f"subgraph_{uuid.uuid4().hex[:4]}")
            # Optionally, remove the temporary file.
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            
            # Use GraphTransformer to add an abstraction node from the repository.
            transformer = GraphTransformer(state["composer"])
            transformer.add_abstraction_node(
                new_subgraph_node,
                chosen_subset=["input"],
                outputs=["output"]
            )
            # Store in the repository.
            repo_entry = {
                "subgraph_node": new_subgraph_node,
                "performance": state["performance"],
                "graph_actions": state["graph_actions"].copy()
            }
            self.repository.append(repo_entry)
            print(f"Repository updated: New best performance {self.best_mse}")

    def get_training_data(self):
        """
        Dummy function to generate training data.
        Replace with actual data from your curriculum.
        """
        import numpy as np
        X = np.random.rand(100, self.input_dim)
        # For an autoencoder, target is the input itself, or you could transform it using the reference encoder.
        y = X.copy()
        return X, y

    def is_terminal(self, state):
        # Define termination criteria (e.g., performance threshold or search budget).
        return False
