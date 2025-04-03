# agents/mcts.py
from agents.mcts_agent_interface import MCTSAgentInterface
from data_gen.curriculum import Curriculum
from utils.nn import create_minimal_graphmodel
from graph.composer import GraphComposer
from graph.node import SingleNeuron
import uuid

class SimpleMCTSAgent(MCTSAgentInterface):
    def __init__(self):
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.input_dim = self.reference['config']['input_dim']
        self.latent_dim = self.reference['config']['encoder'][-1]  # Get the encoder's latent dimension

    def get_initial_state(self):
        # Use latent_dim as output_units, and choose an appropriate activation function (e.g., 'relu').
        composer, _ = create_minimal_graphmodel((self.input_dim,), output_units=self.latent_dim, activation="relu")
        return {
            "composer": composer,
            "graph_actions": [],
            "performance": 1.0,  # dummy value until evaluation is hooked up
            "target_mse": self.target_mse,
        }

    def get_available_actions(self, state):
        return ["add_neuron"]  # We'll start with just this one action

    def apply_action(self, state, action):
        composer = state["composer"]
        new_node = SingleNeuron(name=str(uuid.uuid4()), activation="relu")

        # Connect new neuron between input and output (for now)
        input_node = composer.input_node
        output_node = composer.output_node

        # Disconnect input → output
        composer.disconnect(input_node, output_node)

        # Add new node and wire it: input → new → output
        composer.add_node(new_node)
        composer.connect(input_node, new_node)
        composer.connect(new_node, output_node)

        return {
            "composer": composer,
            "graph_actions": state["graph_actions"] + [action],
            "performance": 1.0,  # still dummy
            "target_mse": state["target_mse"]
        }

    def evaluate_state(self, state):
        return 0.0  # We'll update this in Stage 2

    def is_terminal(self, state):
        return False  # Also to be updated later
