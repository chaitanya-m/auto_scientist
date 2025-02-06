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
    def __init__(self, total_steps=100, num_instances_per_step=100, seed=0):
        """
        total_steps: Total steps (interactions) in the episode.
        num_instances_per_step: Number of data points provided at each step.
        seed: Seed for generating the fixed data distribution.
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.num_instances_per_step = num_instances_per_step
        self.seed = seed

        # Fixed dataset from our classification data generator.
        self.factory = DataSchemaFactory()
        self.schema = self.factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=self.seed
        )
        self.dataset = self.schema.generate_dataset(
            num_instances=self.num_instances_per_step,
            random_seed=123
        )
        # Convert features to numpy array and labels to int.
        self.features = self.dataset[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        self.true_labels = self.dataset["label"].to_numpy(dtype=int)

        # Initialize networks for two agents (agent IDs 0 and 1).
        self.agents_networks = {}
        self.agents_networks[0] = create_minimal_network(input_shape=(2,))
        self.agents_networks[1] = create_minimal_network(input_shape=(2,))

        # Running average accuracy for each agent.
        self.agent_cum_acc = {0: 0.0, 1: 0.0}
        self.agent_steps = {0: 0, 1: 0}

        # Global repository for learned abstractions.
        self.repository = {}

    def reset(self):
        self.current_step = 0
        self.agent_cum_acc = {0: 0.0, 1: 0.0}
        self.agent_steps = {0: 0, 1: 0}
        return self._get_state()

    def _get_state(self):
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
        # Valid actions: "add_abstraction", "no_change".
        return ["add_abstraction", "no_change"]

    def train_subgraph_model(self):
        """
        Builds a learned abstraction: a hidden layer with 3 neurons.
        Uses Keras functionality via our node and composer modules.
        """
        import keras
        from keras import layers, initializers
        # Use a fixed initializer.
        kernel_init = initializers.GlorotUniform(seed=42)
        input_shape = (2,)
        new_input = layers.Input(shape=input_shape, name="sub_input")
        x = layers.Dense(3, activation='relu', name="hidden_layer", kernel_initializer=kernel_init)(new_input)
        model = keras.models.Model(new_input, x, name="learned_abstraction_model")
        subgraph_node = SubGraphNode(name="learned_abstraction", model=model)
        return subgraph_node

    def evaluate_learned_abstraction(self):
        """
        Runs the learned abstraction model on the environment's features and returns the accuracy.
        This method assumes that the learned abstraction is stored in the repository.
        """
        if "learned_abstraction" not in self.repository:
            raise ValueError("Learned abstraction not found in repository.")
        model = self.repository["learned_abstraction"].model
        predictions = model.predict(self.features, verbose=0)
        # For simplicity, assume a sigmoid output: threshold at 0.5.
        preds = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(preds == self.true_labels)
        return accuracy

    def run_model_accuracy(self, agent_id):
        """
        Runs the agent's network on the dataset to compute accuracy.
        """
        _, model = self.agents_networks[agent_id]
        predictions = model.predict(self.features, verbose=0)
        preds = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(preds == self.true_labels)
        return accuracy

    def step(self, actions):
        # Process each agent's action.
        for agent_id, action in actions.items():
            composer, _ = self.agents_networks[agent_id]
            if action == "add_abstraction":
                if "learned_abstraction" in self.repository:
                    learned_node = self.repository["learned_abstraction"]
                    if learned_node.name not in composer.nodes:
                        composer.add_node(learned_node)
                        composer.connect("input", learned_node.name)
                        composer.connect(learned_node.name, "output")
                        self.agents_networks[agent_id] = (composer, composer.build())
            elif action == "no_change":
                pass
            else:
                raise ValueError("Invalid action provided.")

        # Compute accuracies.
        accuracies = {}
        for agent_id in [0, 1]:
            acc = self.run_model_accuracy(agent_id)
            accuracies[agent_id] = acc
            self.agent_steps[agent_id] += 1
            self.agent_cum_acc[agent_id] += acc
        avg_acc = {agent_id: self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
                   for agent_id in [0, 1]}

        rewards = {}
        for agent_id in [0, 1]:
            if accuracies[agent_id] > avg_acc[agent_id]:
                rewards[agent_id] = 1
            elif np.isclose(accuracies[agent_id], avg_acc[agent_id]):
                rewards[agent_id] = 0
            else:
                rewards[agent_id] = -1

        diff = accuracies[0] - accuracies[1]
        if diff > 0:
            rewards[0] += diff * 10
        elif diff < 0:
            rewards[1] += (-diff) * 10

        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.total_steps
        return next_state, rewards, done

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

def run_episode(env, agent0, agent1):
    state = env.reset()
    rewards_history = {0: [], 1: []}
    accuracies_history = {0: [], 1: []}
    while True:
        valid = env.valid_actions()
        actions = {
            0: agent0.choose_action(0, state, valid),
            1: agent1.choose_action(1, state, valid)
        }
        state, rewards, done = env.step(actions)
        rewards_history[0].append(rewards[0])
        rewards_history[1].append(rewards[1])
        acc0 = env.run_model_accuracy(0)
        acc1 = env.run_model_accuracy(1)
        accuracies_history[0].append(acc0)
        accuracies_history[1].append(acc1)
        if done:
            break
    # Return a dictionary of actions histories, rewards history, and accuracies history.
    return {0: agent0.actions_history[0], 1: agent0.actions_history[1]}, rewards_history, accuracies_history

