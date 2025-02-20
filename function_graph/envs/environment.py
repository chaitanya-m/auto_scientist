# envs/environment.py
import numpy as np
from abc import ABC, abstractmethod
from utils.nn import create_minimal_graphmodel

# -----------------------
# Environment Interface
# -----------------------

class Environment(ABC):
    def __init__(self, transition_rule, reward_rule):
        """
        A minimal state transition environment interface.

        :param transition_rule: A callable that, given the current state and a dictionary of actions,
                                returns the next state.
        :param reward_rule: A callable that, given the current state, actions, and next state,
                            computes and returns a reward (or dictionary of rewards).
        """
        self.transition_rule = transition_rule
        self.reward_rule = reward_rule
        self.state = None  # No initial state is generated automatically.

    @abstractmethod
    def reset(self, initial_state):
        """
        Resets the environment to the given initial state.

        :param initial_state: The initial State object.
        :return: The initial state.
        """
        pass

    @abstractmethod
    def step(self, actions):
        """
        Applies the transition rule to update the state.

        :param actions: A dictionary mapping agent IDs to their actions.
        :return: A tuple (next_state, reward, done).
        """
        pass

# -----------------------
# Legacy Multi-Agent Environment (implements Environment)
# -----------------------

class AgentState:
    def __init__(self, nodes, connections, graphmodel):
        """
        Represents the state of an individual agent.
        
        :param nodes: List of node identifiers from the agent's composer.
        :param connections: The connection structure from the agent's composer.
        :param graphmodel: The agent's current graphmodel (e.g., a tuple of (composer, model)).
        """
        self.nodes = nodes
        self.connections = connections
        self.graphmodel = graphmodel

    def __repr__(self):
        return f"AgentState(nodes={self.nodes}, connections={self.connections})"
        

class State:
    def __init__(self, dataset, agents_states):
        """
        Represents the overall environment state.

        :param dataset: The dataset generated at the current step.
        :param agents_states: A dictionary mapping agent IDs to their AgentState.
                              Each AgentState contains details about the agent's nodes,
                              connections, and current graphmodel.
        """
        self.dataset = dataset
        self.agents_states = agents_states

    def __repr__(self):
        return f"State(dataset={self.dataset}, agents_states={self.agents_states})"



class RLEnvironment(Environment):
    def __init__(self, num_instances_per_step=100, seed=0, n_agents=2, schema=None):
        """
        Legacy RLEnvironment that manages multi-agent setups, data generation, and reward computation.

        :param num_instances_per_step: Number of data points provided at each step.
        :param seed: Seed for generating the fixed data distribution.
        :param n_agents: Number of agents in the environment.
        :param schema: A data schema object used for generating datasets.
        """
        # For the legacy environment, we don't need transition_rule and reward_rule at initialization.
        # We will provide default (dummy) callables.
        super().__init__(transition_rule=lambda state, actions: self._default_transition(state, actions),
                         reward_rule=lambda state, actions, next_state: self.compute_rewards_default(state, actions, next_state))
        
        self.num_instances_per_step = num_instances_per_step
        self.seed = seed
        
        # Environment dataset
        self.schema = schema
        self.dataset = None

        # Create agent graphmodels and initialize reward trackers.
        self.n_agents = n_agents
        self.agents_graphmodels = {}
        self.agent_cum_acc = {}
        self.agent_steps = {}
        for agent_id in range(self.n_agents):
            self.agents_graphmodels[agent_id] = create_minimal_graphmodel(input_shape=(2,))
            self.agent_cum_acc[agent_id] = 0.0
            self.agent_steps[agent_id] = 0

        self.repository = {}

    def reset(self, initial_state=None, seed: int = None, new_schema=None):
        """
        Resets the environment state. For legacy behavior, an initial dataset can be generated via schema.
        Either supply a new_schema or a new seed.
        """
        if new_schema is None and seed is None:
            raise ValueError("Either a new_schema or a seed must be provided for reset()")
        if new_schema is not None:
            self.schema = new_schema
        else:
            self.schema.rng = np.random.default_rng(seed)
        # Generate an initial dataset if none is provided.
        if initial_state is None:
            self.dataset = self.schema.generate_dataset(num_instances=self.num_instances_per_step)
            initial_state = self._get_state()
        self.state = initial_state
        return self.state

    def _get_state(self):
        """
        Constructs and returns the current state as a State object.
        """
        agents_states = {}
        for agent_id, graphmodel in self.agents_graphmodels.items():
            # Extract composer and connections from the graphmodel.
            composer, _ = graphmodel
            agent_state = AgentState(
                nodes=list(composer.nodes.keys()),
                connections=composer.connections,
                graphmodel=graphmodel
            )
            agents_states[agent_id] = agent_state

        return State(dataset=self.dataset, agents_states=agents_states)

    def step(self, actions=None):
        """
        Generates new data for this step and returns the updated state.
        Actions are ignored in the legacy environment's default transition.
        """
        self.dataset = self.schema.generate_dataset(num_instances=self.num_instances_per_step)
        state = self._get_state()
        # For legacy behavior, 'actions' are not used to update the state.
        return state

    def compute_rewards_default(self, state, actions, next_state):
        """
        Default reward computation for the legacy environment.
        """
        # Update running average accuracy.
        dummy_accuracies = {}  # This function would normally use some computed accuracies.
        for agent_id in range(self.n_agents):
            # For simplicity, assume a dummy accuracy value (e.g., 1.0) for demonstration.
            acc = 1.0
            self.agent_steps[agent_id] += 1
            self.agent_cum_acc[agent_id] += acc
            dummy_accuracies[agent_id] = acc
        
        rewards = {}
        for agent_id, acc in dummy_accuracies.items():
            avg_acc = self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
            rewards[agent_id] = avg_acc
        return rewards

    def compute_rewards(self, accuracies):
        """
        Computes rewards based on provided accuracies.
        """
        for agent_id, acc in accuracies.items():
            self.agent_steps[agent_id] += 1
            self.agent_cum_acc[agent_id] += acc
        rewards = {}
        for agent_id, acc in accuracies.items():
            avg_acc = self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
            rewards[agent_id] = avg_acc
        return rewards

    def _default_transition(self, state, actions):
        """
        A dummy transition rule that simply returns the current state.
        """
        return state
