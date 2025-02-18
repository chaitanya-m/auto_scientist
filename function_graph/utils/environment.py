#rl_dev.py
import numpy as np
import pandas as pd
from data_gen.categorical_classification import DataSchemaFactory
from graph.node import InputNode, SingleNeuron, SubGraphNode
from graph.composer import GraphComposer, GraphTransformer

# -----------------------
# Environment for multiple agents
# -----------------------

def create_minimal_network(input_shape):
    """
    Builds a minimal valid network using our node and composer framework.
    The network has an InputNode (with the given input_shape) and a SingleNeuron output node with sigmoid activation.
    """
    composer = GraphComposer()
    input_node = InputNode(name="input", input_shape=input_shape)
    # For binary classification, we use a sigmoid on the output.
    output_node = SingleNeuron(name="output", activation="sigmoid")
    composer.add_node(input_node)
    composer.add_node(output_node)
    composer.set_input_node("input")
    composer.set_output_node("output")
    composer.connect("input", "output")
    model = composer.build()
    return composer, model


import numpy as np

# Assume these functions/classes are defined elsewhere:
# DataSchemaFactory, create_minimal_network

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
    def __init__(self, step, dataset, agents_states):
        """
        Represents the overall environment state.
        
        :param step: The current step in the environment.
        :param dataset: The dataset generated at the current step.
        :param agents_states: A dictionary mapping external agent IDs to their AgentState.
        """
        self.step = step
        self.dataset = dataset
        self.agents_states = agents_states

    def __repr__(self):
        return f"State(step={self.step}, dataset=DataFrame(...), agents_states={self.agents_states})"


class RLEnvironment:
    def __init__(self, total_steps=100, num_instances_per_step=100, seed=0, n_agents=2):
        """
        total_steps: Total steps (interactions) in the episode.
        num_instances_per_step: Number of data points provided at each step.
        seed: Seed for generating the fixed data distribution.
        n_agents: Number of agents in the environment (default 2).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.num_instances_per_step = num_instances_per_step
        self.seed = seed

        # Create the schema once with a fixed seed so distribution parameters are fixed.
        self.factory = DataSchemaFactory()
        self.schema = self.factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=self.seed
        )
        
        # Initialize environment data placeholders.
        self.dataset = None
        self.features = None
        self.true_labels = None

        # Dynamically create networks for n_agents rather than hardcoding 0 and 1.
        self.n_agents = n_agents
        self.agents_networks = {}
        self.agent_cum_acc = {}
        self.agent_steps = {}
        for agent_id in range(self.n_agents):
            self.agents_networks[agent_id] = create_minimal_network(input_shape=(2,))
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
        self.current_step = 0
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
        for agent_id, (composer, _) in self.agents_networks.items():
            agent_state = AgentState(
                nodes=list(composer.nodes.keys()),
                connections=composer.connections
            )
            agents_states[agent_id] = agent_state
        return State(
            step=self.current_step,
            dataset=self.dataset,
            agents_states=agents_states
        )

    def valid_actions(self):
        # If you add new actions, just include them here.
        return ["add_abstraction", "no_change"]

    def step(self):
        # Generate new data for this step.
        self.dataset = self.schema.generate_dataset(num_instances=self.num_instances_per_step)
        # Optionally store features/labels for tests that need them.
        self.features = self.dataset[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        self.true_labels = self.dataset["label"].to_numpy(dtype=int)
        
        self.current_step += 1
        state = self._get_state()
        done = self.current_step >= self.total_steps
        return state, done

    def compute_rewards(self, accuracies):
        """
        Update each agent's cumulative accuracy and assign base reward.
        If exactly two agents are present, apply difference-based bonus/penalty.
        """
        # Update running average accuracy.
        for agent_id, acc in accuracies.items():
            self.agent_steps[agent_id] += 1
            self.agent_cum_acc[agent_id] += acc
        
        rewards = {}
        for agent_id, acc in accuracies.items():
            avg_acc = self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
            if acc > avg_acc:
                rewards[agent_id] = 1
            elif np.isclose(acc, avg_acc):
                rewards[agent_id] = 0
            else:
                rewards[agent_id] = -1

        # If exactly 2 agents, replicate the old difference-based bonus approach.
        if self.n_agents == 2:
            agent_list = list(self.agents_networks.keys())
            agent_a, agent_b = agent_list[0], agent_list[1]
            diff = accuracies[agent_a] - accuracies[agent_b]
            if diff > 0:
                rewards[agent_a] += diff * 10
            elif diff < 0:
                rewards[agent_b] += (-diff) * 10

        return rewards


def split_dataset(dataset):
    split_idx = len(dataset) // 2
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]
    return train_df, test_df




def run_episode(env, agents, seed=0, schema=None):
    """
    Runs an episode of the environment for exactly env.total_steps steps,
    for one or more agents. Each agent can choose actions (including
    "add_abstraction") at each step.

    Parameters
    ----------
    env : RLEnvironment
        The environment instance, which must contain a repository
        with any learned abstraction(s) for "add_abstraction" to work.
    agents : dict[int, DummyAgent or similar]
        A dictionary mapping agent IDs to agent instances that have
        choose_action(...) and evaluate_accuracy(...).
    seed : int, optional
        Seed for resetting the environment, by default 0.
    schema : optional
        Optional new schema for environment reset, by default None.

    Returns
    -------
    actions_history : dict[int, list[str]]
        The sequence of actions each agent took, keyed by agent ID.
    rewards_history : dict[int, list[float]]
        The sequence of rewards each agent received over env.total_steps, keyed by agent ID.
    accuracies_history : dict[int, list[float]]
        The sequence of accuracies each agent achieved at each step, keyed by agent ID.
    """
    # 1. Reset environment to a fresh state. Note that the returned state is a State instance.
    state = env.reset(seed=seed, new_schema=schema)
    
    # 2. Compile models for each agent if needed.
    for agent_id, (composer, model) in env.agents_networks.items():
        if not hasattr(model, "optimizer"):
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Prepare history dictionaries.
    actions_history = {agent_id: [] for agent_id in env.agents_networks}
    rewards_history = {agent_id: [] for agent_id in env.agents_networks}
    accuracies_history = {agent_id: [] for agent_id in env.agents_networks}
    debug_counter = {agent_id: 0 for agent_id in env.agents_networks}
    
    # 3. Main loop over total_steps.
    for _ in range(env.total_steps):
        # Generate new data and update the environment state.
        state, done = env.step()

        # Each agent picks an action based on its AgentState from the overall State object.
        chosen_actions = {}
        valid = env.valid_actions()
        for agent_id in agents:
            # Use the AgentState from the new State object.
            agent_state = state.agents_states[agent_id]
            action = agents[agent_id].choose_action(agent_state, valid)
            chosen_actions[agent_id] = action
            actions_history[agent_id].append(action)

        # 4. Apply any "add_abstraction" actions.
        for agent_id, action in chosen_actions.items():
            if action == "add_abstraction":
                debug_counter[agent_id] += 1
                print(f"DEBUG: {agent_id} adds abstraction {debug_counter[agent_id]} times")
                learned_abstraction = env.repository["learned_abstraction"]
                composer, model = env.agents_networks[agent_id]
                transformer = GraphTransformer(composer)
                
                # For simplicity, always connect "input" -> abstraction -> "output", removing direct input->output.
                new_model = transformer.add_abstraction_node(
                    abstraction_node=learned_abstraction,
                    chosen_subset=["input"],
                    outputs=["output"],
                    remove_prob=1.0
                )
                env.agents_networks[agent_id] = (composer, new_model)

        # 5. Each agent trains/evaluates on the newly generated dataset.
        accuracies = {}
        for agent_id in agents:
            _, model = env.agents_networks[agent_id]
            acc = agents[agent_id].evaluate_accuracy(model, env.dataset)
            accuracies_history[agent_id].append(acc)
            accuracies[agent_id] = acc

        # 6. Compute rewards using the environment's logic.
        rewards = env.compute_rewards(accuracies)
        for agent_id in agents:
            rewards_history[agent_id].append(rewards[agent_id])
        
        if done:
            break

    return actions_history, rewards_history, accuracies_history


# -----------------------
# Dummy Agent
# -----------------------
class DummyAgent:
    def __init__(self, action_plan=None):
        self.actions_history = {0: [], 1: []}
        self.action_plan = action_plan or {0: [], 1: []}

    def choose_action(self, agent_id, state, valid_actions):
        if self.action_plan[agent_id]:
            action = self.action_plan[agent_id].pop(0)
        else:
            action = np.random.choice(valid_actions)
        self.actions_history[agent_id].append(action)
        return action

    def evaluate_accuracy(self, model, dataset):
            # Assume dataset is a DataFrame.
            split_idx = len(dataset) // 2
            train_df = dataset.iloc[:split_idx]
            test_df = dataset.iloc[split_idx:]
            
            # Extract features and labels for training.
            train_features = train_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
            train_labels = train_df["label"].to_numpy(dtype=int)
            # Extract features and labels for testing.
            test_features = test_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
            test_labels = test_df["label"].to_numpy(dtype=int)
            
            # Optionally train/fine-tune the model on training data.
            model.fit(train_features, train_labels, epochs=1, verbose=0)
            
            # Evaluate on test data.
            predictions = model.predict(test_features, verbose=0)
            preds = (predictions.flatten() > 0.5).astype(int)
            accuracy = np.mean(preds == test_labels)
            return accuracy
