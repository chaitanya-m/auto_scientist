import numpy as np
import pandas as pd
from data_gen.categorical_classification import DataSchemaFactory
from graph.node import InputNode, SingleNeuron
from graph.composer import GraphComposer

# -----------------------
# Minimal Network Construction
# -----------------------
def create_minimal_network(input_shape):
    """
    Builds a minimal valid network using our node and composer framework.
    The network has an InputNode and an output SingleNeuron.
    """
    composer = GraphComposer()
    # Create nodes
    input_node = InputNode(name="input", input_shape=input_shape)
    output_node = SingleNeuron(name="output", activation=None)
    # Add nodes to composer
    composer.add_node(input_node)
    composer.add_node(output_node)
    # Define the network: input -> output
    composer.set_input_node("input")
    composer.set_output_node("output")
    composer.connect("input", "output")
    # Build the Keras model
    model = composer.build()
    return composer, model

# -----------------------
# RL Environment
# -----------------------
class RLEnvironment:
    def __init__(self, total_steps=1000, num_instances_per_step=100, seed=0):
        """
        total_steps: Total number of steps (interactions) in the episode.
        num_instances_per_step: Number of data points provided at each step.
        seed: Seed for generating the fixed data distribution.
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.num_instances_per_step = num_instances_per_step
        self.seed = seed

        # Create the fixed classification dataset using our data generator.
        self.factory = DataSchemaFactory()
        self.schema = self.factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=self.seed
        )
        self.dataset = self.schema.generate_dataset(
            num_instances=self.num_instances_per_step,
            random_seed=123  # fixed seed for dataset generation within the episode
        )
        # Create the minimal valid network.
        # The input shape matches the number of features (2 in this case).
        self.composer, self.model = create_minimal_network(input_shape=(2,))

    def reset(self):
        self.current_step = 0
        # In a full implementation, you might reset the network state as well.
        return self._get_state()

    def _get_state(self):
        # The state includes the current step, the fixed dataset, and a summary of the network.
        # For simplicity, we provide the network's current node names.
        network_summary = {
            "nodes": list(self.composer.nodes.keys()),
            "connections": self.composer.connections
        }
        return {
            "step": self.current_step,
            "dataset": self.dataset,
            "network": network_summary
        }

    def valid_actions(self):
        # Initially, the valid actions are to "reuse" a component from the repository or add a "new" node.
        # In later stages, more nuanced actions might be available.
        return ["reuse", "new"]

    def step(self, action):
        """
        At each step, the environment provides the same dataset and the current network.
        The action will eventually be used to modify the network.
        Here we simply return the current state, a dummy reward, and whether the episode is done.
        """
        # For now, ignore the effect of the action on the network.
        reward = 0  # Placeholder for reward calculation.
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.total_steps
        return next_state, reward, done

# -----------------------
# Dummy Agent
# -----------------------
class DummyAgent:
    def __init__(self):
        self.actions_history = []

    def choose_action(self, state, valid_actions):
        # For demonstration, choose randomly.
        action = np.random.choice(valid_actions)
        self.actions_history.append(action)
        return action

def run_episode(env, agent):
    state = env.reset()
    rewards = []
    while True:
        actions = env.valid_actions()
        action = agent.choose_action(state, actions)
        state, reward, done = env.step(action)
        rewards.append(reward)
        if done:
            break
    return agent.actions_history, rewards

# -----------------------
# Example Execution
# -----------------------
if __name__ == "__main__":
    env = RLEnvironment(total_steps=1000, num_instances_per_step=100, seed=0)
    agent = DummyAgent()
    actions, rewards = run_episode(env, agent)
    print(f"Total steps: {env.current_step}")
    print(f"Actions taken (first 10): {actions[:10]} ...")
    # Display a sample of the dataset and current network summary.
    state = env._get_state()
    print("Dataset sample:")
    print(state["dataset"].head())
    print("Network summary:")
    print(state["network"])
