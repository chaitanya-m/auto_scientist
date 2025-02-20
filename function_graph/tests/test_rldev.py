import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # Force CPU usage.
os.environ["TF_DETERMINISTIC_OPS"] = "1"          # Request deterministic operations.

import random
import numpy as np
import tensorflow as tf
import unittest

# Seed all relevant random generators.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import keras
from keras import layers, initializers, utils
from graph.node import SubGraphNode
from envs.environment import RLEnvironment
from utils.nn import create_minimal_graphmodel, train_learned_abstraction_model
from utils.visualize import visualize_graph, print_graph_nodes
from agents.deterministic import DeterministicAgent
from graphviz import Digraph    
from data_gen.categorical_classification import DataSchemaFactory
from utils.rl import run_episode


class TestLearnedAbstractionTraining(unittest.TestCase):
    def test_learned_abstraction_training(self):
        """
        This test:
          1. Creates a minimal network with one input node and one output node (with sigmoid activation).
          2. Uses a helper to train a learned abstraction (a hidden layer with 3 neurons) on a dataset of 500 examples.
          3. Adds the learned abstraction into the network by connecting input -> abstraction -> output.
          4. Freezes the learned abstraction so its weights remain unchanged.
          5. Saves the learned abstraction's weights.
          6. Rebuilds the Keras model via the composer.
          7. Fine-tunes the composed model for 300 epochs (only the output neuron's weights should change).
          8. Evaluates and prints the final accuracy.
          9. Verifies that the learned abstraction's weights are identical before and after fine-tuning.
          10. Stores the learned abstraction in the repository.
          11. Asserts that the final accuracy is above a chosen threshold.
        """

        # Create the schema once with a fixed seed so distribution parameters are fixed.
        factory = DataSchemaFactory()
        schema = factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=SEED
        )

        # Create the environment with 500 examples per step.
        env = RLEnvironment(num_instances_per_step=500, seed=0, n_agents=1, schema=schema)
        state = env.reset(seed=0)

        # For this test, we directly extract the dataset.
        dataset = state.dataset

        # Build the minimal network.
        composer, model = create_minimal_graphmodel(input_shape=(2,))
        
        # Train the learned abstraction on 500 examples.
        learned_abstraction = train_learned_abstraction_model(dataset, epochs=1000)
        
        # Freeze the learned abstraction so its weights remain unchanged.
        learned_abstraction.model.trainable = False
        weights_before = learned_abstraction.model.get_weights()

        # Add the learned abstraction into the network.
        composer.add_node(learned_abstraction)
        composer.connect("input", learned_abstraction.name)
        composer.connect(learned_abstraction.name, "output")
        composer.remove_connection("input", "output")

        # Rebuild the Keras model.
        model = composer.build()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Optionally visualize the graph.
        # print_graph_nodes(composer)
        # visualize_graph(composer=composer)

        # Fine-tune the composed model.
        features = dataset[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
        true_labels = dataset["label"].to_numpy(dtype=int)
        model.fit(features, true_labels, epochs=300, verbose=0)

        # Evaluate the fine-tuned model.
        loss, acc = model.evaluate(features, true_labels, verbose=0)
        print(f"Trained network accuracy after fine-tuning: {acc:.3f}")
        
        # Check that the learned abstraction's weights have not changed.
        weights_after = learned_abstraction.model.get_weights()
        for w_before, w_after in zip(weights_before, weights_after):
            assertion = "Learned abstraction weights should remain unchanged after fine-tuning."
            np.testing.assert_allclose(w_before, w_after, atol=1e-7, err_msg=assertion)
            print(assertion)
        
        # Store the learned abstraction in the repository.
        env.repository["learned_abstraction"] = learned_abstraction
        
        # Assert final accuracy is above threshold.
        assertion = "Trained network accuracy should be above 0.9."
        print(assertion)
        self.assertGreater(acc, 0.9, assertion)


def my_custom_policy(state, step, valid_actions):
    """
    A custom policy that tries to pick 'add_abstraction' at step 0 if it's
    in the valid_actions list, otherwise falls back to selecting 'no_change'
    or a random valid action.
    """
    import numpy as np
    
    if step == 0 and "add_abstraction" in valid_actions:
        return "add_abstraction"
    elif "no_change" in valid_actions:
        return "no_change"
    else:
        return np.random.choice(valid_actions) if valid_actions else None


class TestReuseAdvantage(unittest.TestCase):
    def test_reuse_advantage(self):
        """
        Hypothesis: When one agent adds the learned abstraction and the other does not,
        the agent that adds it will obtain higher accuracy and receive a reward advantage.
        
        Test: Pre-populate the repository with the learned abstraction.
        Run a controlled number of steps where agent 0 chooses "add_abstraction" on the first step,
        while agent 1 always chooses "no_change". Print detailed step-by-step accuracies and rewards,
        and verify that agent 0's cumulative reward exceeds that of agent 1.
        """
        num_steps = 5

        # Create a schema externally and pass it to the environment.
        factory = DataSchemaFactory()
        schema = factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=0
        )

        # Create the environment with the externally provided schema.
        env = RLEnvironment(num_instances_per_step=100, seed=0, n_agents=2, schema=schema)
        
        # Reset environment and generate a dataset to train the abstraction.
        state = env.reset(seed=0)
        dataset = state.dataset

        # Train an abstraction on the dataset and store it in the environment's repository.
        learned_abstraction = train_learned_abstraction_model(dataset, epochs=1000)
        env.repository["learned_abstraction"] = learned_abstraction

        # Create DeterministicAgents.
        # Agent 0 uses a custom policy to choose "add_abstraction" on step 0.
        agent0 = DeterministicAgent(policy=my_custom_policy, training_params={"epochs":300, "verbose":0})
        agent0.update_valid_actions(["add_abstraction", "no_change"])

        # Agent 1 always chooses "no_change".
        agent1 = DeterministicAgent(training_params={"epochs":300, "verbose":0})
        agent1.update_valid_actions(["no_change"])

        # Prepare the dictionary of agents.
        agents_dict = {
            0: agent0,
            1: agent1
        }

        # Run the episode for the specified number of steps.
        actions, rewards, accuracies = run_episode(env, agents_dict, seed=0, steps=num_steps, schema=schema, step_callback=None)

        print("\nDetailed Step-by-Step Output:")
        for step in range(num_steps):
            print(f"Step {step+1}:")
            print(f"  Agent 0: Accuracy = {accuracies[0][step]:.3f}, Reward = {rewards[0][step]:.3f}")
            print(f"  Agent 1: Accuracy = {accuracies[1][step]:.3f}, Reward = {rewards[1][step]:.3f}")

        # Compute cumulative rewards.
        reward0 = sum(rewards[0])
        reward1 = sum(rewards[1])
 
        diff = reward0 - reward1
        print(f"\nTest outcome: Agent 0's reward = {reward0}, Agent 1's reward = {reward1}.")
        print(f"Agent 0 won by a margin of {diff} reward points.")
        print("Actions taken:", actions)
        
        self.assertGreater(
            reward0,
            reward1,
            "Agent 0 should receive a higher reward than Agent 1 when using the learned abstraction."
        )

if __name__ == "__main__":
    unittest.main()
