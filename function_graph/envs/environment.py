# utils/environment.py
import numpy as np
from utils.nn import create_minimal_graphmodel


# -----------------------
# Environment for multiple agents
# -----------------------


class AgentState:
    def __init__(self, nodes, connections):
        """
        Represents the state of an individual agent.
        
        :param nodes: List of node identifiers from the agent's composer.
        :param connections: The connection structure from the agent's composer.
        """
        self.nodes = nodes
        self.connections = connections

    def __repr__(self):
        return f"AgentState(nodes={self.nodes}, connections={self.connections})"


class State:
    def __init__(self, dataset, agents_states):
        """
        Represents the overall environment state.
        
        :param step: The current step in the environment.
        :param dataset: The dataset generated at the current step.
        :param agents_states: A dictionary mapping external agent IDs to their AgentState.
        """
        self.dataset = dataset
        self.agents_states = agents_states

    def __repr__(self):
        return f"State(agents_states={self.agents_states})"


class RLEnvironment:
    def __init__(self, num_instances_per_step=100, seed=0, n_agents=2, schema=None):
        """
        total_steps: Total steps (interactions) in the episode.
        num_instances_per_step: Number of data points provided at each step.
        seed: Seed for generating the fixed data distribution.
        n_agents: Number of agents in the environment (default 2).
        """

        self.num_instances_per_step = num_instances_per_step
        self.seed = seed
        
        # Environment dataset
        self.schema = schema
        self.dataset = None

        # Dynamically create networks for n_agents rather than hardcoding 0 and 1.
        self.n_agents = n_agents
        self.agents_graphmodels = {}
        self.agent_cum_acc = {}
        self.agent_steps = {}
        for agent_id in range(self.n_agents):
            self.agents_graphmodels[agent_id] = create_minimal_graphmodel(input_shape=(2,))
            self.agent_cum_acc[agent_id] = 0.0
            self.agent_steps[agent_id] = 0

        # Global repository for learned abstractions.
        self.repository = {}

    def reset(self, seed: int = None, new_schema=None):
        """
        Resets the environment state.
        Either supply a new_schema or a new seed to reinitialize the distribution.
        """
        if new_schema is None and seed is None:
            raise ValueError("Either a new_schema or a seed must be provided for reset()")

        if new_schema is not None:
            self.schema = new_schema
        else:
            self.schema.rng = np.random.default_rng(seed)
        return self._get_state()

    def _get_state(self):
        """
        Constructs and returns the current environment state as a State object,
        with each agent's state represented by an AgentState instance.
        """
        agents_states = {}
        for agent_id, (composer, _) in self.agents_graphmodels.items():
            agent_state = AgentState(
                nodes=list(composer.nodes.keys()),
                connections=composer.connections
            )
            agents_states[agent_id] = agent_state

        return State(
            dataset=self.dataset,
            agents_states=agents_states
        )

    def step(self):
        # Generate new data for this step.
        self.dataset = self.schema.generate_dataset(num_instances=self.num_instances_per_step)
        state = self._get_state()
        return state

    def compute_rewards(self, accuracies):
        """
        Update each agent's cumulative accuracy and assign base reward.
        """
        # Update running average accuracy.
        for agent_id, acc in accuracies.items():
            self.agent_steps[agent_id] += 1
            self.agent_cum_acc[agent_id] += acc
        
        rewards = {}
        for agent_id, acc in accuracies.items():
            avg_acc = self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
            rewards[agent_id] = avg_acc

        return rewards
