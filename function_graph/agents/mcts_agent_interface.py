from abc import ABC, abstractmethod
from data_gen.curriculum import Curriculum
from graph.composer import GraphComposer
from utils.nn import create_minimal_graphmodel
from graph.node import SingleNeuron
import uuid

class MCTSAgentInterface(ABC):
    @abstractmethod
    def get_initial_state(self): pass

    @abstractmethod
    def get_available_actions(self, state): pass

    @abstractmethod
    def apply_action(self, state, action): pass

    @abstractmethod
    def evaluate_state(self, state): pass

    @abstractmethod
    def is_terminal(self, state): pass


class SimpleMCTSAgent(MCTSAgentInterface):
    def __init__(self):
        self.curriculum = Curriculum(phase_type='basic')
        self.reference = self.curriculum.get_reference(0, seed=0)
        self.target_mse = self.reference['mse']
        self.encoder_config = self.reference['config']['encoder']
        self.input_dim = self.reference['config']['input_dim']
        self.output_dim = self.reference['config']['output_dim']

    def get_initial_state(self):
        composer, _ = create_minimal_graphmodel((self.input_dim,))
        return {
            "composer": composer,
            "graph_actions": [],
            "performance": 1.0,  # dummy until we hook up evaluation
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
