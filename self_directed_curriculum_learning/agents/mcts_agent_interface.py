#agents/mcts_agent_interface.py
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
