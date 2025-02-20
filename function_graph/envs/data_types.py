# -----------------------
# State and AgentState data types
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

    # Shared repository for all State instances.
    repository = {}

    def __init__(self, dataset, agents_states, repository=None):
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
