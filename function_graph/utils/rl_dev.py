#rl_dev.py
import numpy as np
import pandas as pd
from data_gen.categorical_classification import DataSchemaFactory
from graph.node import InputNode, SingleNeuron, SubGraphNode
from graph.composer import GraphComposer

# -----------------------
# Environment for Two Agents
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
        # Build a state dictionary for each agent’s composer connections etc.
        agents_state = {}
        for agent_id, (composer, _) in self.agents_networks.items():
            agents_state[agent_id] = {
                "nodes": list(composer.nodes.keys()),
                "connections": composer.connections
            }
        return {
            "step": self.current_step,
            "dataset": self.dataset,
            "agents_networks": agents_state
        }

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
            # For convenience, just get their IDs from the dict:
            agent_list = list(self.agents_networks.keys())
            agent_a, agent_b = agent_list[0], agent_list[1]
            diff = accuracies[agent_a] - accuracies[agent_b]
            if diff > 0:
                rewards[agent_a] += diff * 10
            elif diff < 0:
                rewards[agent_b] += (-diff) * 10

        return rewards


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

def split_dataset(dataset):
    split_idx = len(dataset) // 2
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]
    return train_df, test_df


def run_episode(env: RLEnvironment, agent0: DummyAgent, agent1: DummyAgent, seed=0, schema=None):
    # Reset the environment and generate an initial state.
    state = env.reset(seed=seed, new_schema=schema)
    
    # Ensure that both agent networks are compiled.
    for agent_id in [0, 1]:
        composer, model = env.agents_networks[agent_id]
        # The simplest check: if model hasn't been compiled at all, compile now.
        if not hasattr(model, "optimizer"):
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    rewards_history = {0: [], 1: []}
    accuracies_history = {0: [], 1: []}
    
    # Run exactly env.total_steps iterations.
    for _ in range(env.total_steps):
        state, _ = env.step()  # Now env.dataset, env.features, env.true_labels are updated.
        
        valid = env.valid_actions()
        
        # Agents choose their actions.
        action0 = agent0.choose_action(0, state, valid)
        action1 = agent1.choose_action(1, state, valid)
        
        # -- Minimal "apply abstraction" logic --
        if action0 == "add_abstraction":
            # Retrieve the learned abstraction from env.repository.
            learned_abstraction = env.repository["learned_abstraction"]
            
            # Insert it into agent 0's composer.
            composer0, model0 = env.agents_networks[0]
            composer0.add_node(learned_abstraction)
            composer0.connect("input", learned_abstraction.name)
            composer0.connect(learned_abstraction.name, "output")
            composer0.remove_connection("input", "output")
            
            # Rebuild & recompile agent 0's updated model.
            model0 = composer0.build()
            model0.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            env.agents_networks[0] = (composer0, model0)
        
        if action1 == "add_abstraction":
            # Same logic, if you want to allow agent 1 to add abstractions
            learned_abstraction = env.repository["learned_abstraction"]
            
            composer1, model1 = env.agents_networks[1]
            composer1.add_node(learned_abstraction)
            composer1.connect("input", learned_abstraction.name)
            composer1.connect(learned_abstraction.name, "output")
            composer1.remove_connection("input", "output")
            
            model1 = composer1.build()
            model1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            env.agents_networks[1] = (composer1, model1)
        
        # Now each agent trains/evaluates on the new dataset.
        acc0 = agent0.evaluate_accuracy(env.agents_networks[0][1], env.dataset)
        acc1 = agent1.evaluate_accuracy(env.agents_networks[1][1], env.dataset)
        accuracies_history[0].append(acc0)
        accuracies_history[1].append(acc1)
        
        # Compute rewards for both agents based on their accuracies.
        rewards = env.compute_rewards({0: acc0, 1: acc1})
        rewards_history[0].append(rewards[0])
        rewards_history[1].append(rewards[1])
    
    return {0: agent0.actions_history[0], 1: agent1.actions_history[1]}, rewards_history, accuracies_history
