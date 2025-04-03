# agents/mcts.py
from agents.mcts_agent_interface import MCTSAgentInterface
from data_gen.curriculum import Curriculum
from utils.nn import create_minimal_graphmodel
from graph.composer import GraphComposer
from graph.node import SingleNeuron
import uuid

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
        # For now, we have two actions: add a neuron or delete a repository entry.
        actions = ["add_neuron", "delete_repository_entry"]
        # Optionally, we can also include actions to add a subgraph from the repository
        if self.repository:
            actions.append("add_from_repository")
        return actions
    
    def apply_action(self, state, action):
        composer = state["composer"]
        new_state = state.copy()
        new_state["graph_actions"] = state["graph_actions"] + [action]
        
        if action == "add_neuron":
            # Add a new neuron (using SingleNeuron for now) to the graph.
            # Here we choose a random identifier for the new node.
            new_node = SingleNeuron(name=str(uuid.uuid4()), activation="relu")
            
            # For simplicity, assume we are adding the neuron between the input and output.
            # First, disconnect the current connection.
            try:
                composer.disconnect("input", "output")
            except Exception:
                # If no connection exists, skip disconnection.
                pass
                
            # Add the new neuron node.
            composer.add_node(new_node)
            # Connect: input -> new neuron -> output.
            composer.connect("input", new_node.name)
            composer.connect(new_node.name, "output")
        
        elif action == "delete_repository_entry":
            # For now, we simply remove the last entry in the repository.
            if self.repository:
                self.repository.pop()
                
        elif action == "add_from_repository":
            # Dummy action: if repository is not empty, add its first element to the graph.
            # In a more advanced version, you'd merge the repository's subgraph intelligently.
            repo_entry = self.repository[0]
            # Assume repo_entry contains a composer or subgraph that can be merged.
            # For now, we just simulate this by adding a neuron with a different activation.
            new_node = SingleNeuron(name=str(uuid.uuid4()), activation="tanh")
            try:
                composer.disconnect("input", "output")
            except Exception:
                pass
            composer.add_node(new_node)
            composer.connect("input", new_node.name)
            composer.connect(new_node.name, "output")
        
        # Clear any built model so that a new one is constructed on the next evaluation.
        composer.keras_model = None
        # Rebuild the model.
        composer.build()
        
        # Dummy update: for now, performance remains unchanged.
        new_state["performance"] = state["performance"]
        return new_state

    def evaluate_state(self, state):
        # Dummy evaluation: for now, we simply return a dummy performance value.
        # In a real scenario, you'd train the model briefly and compute its MSE.
        return state["performance"]

    def is_terminal(self, state):
        # Dummy terminal check: we'll always return False until we add proper criteria.
        return False
