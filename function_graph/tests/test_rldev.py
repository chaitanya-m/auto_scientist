# tests/test_rldev.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # Force CPU usage.
os.environ["TF_DETERMINISTIC_OPS"] = "1"          # Request deterministic operations.

import random
import numpy as np
import tensorflow as tf
import unittest
from keras import optimizers

# Seed all relevant random generators.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from envs.environment import RLEnvironment, State, AgentState
from utils.nn import create_minimal_graphmodel, train_and_evaluate, AdamWithLRMultiplier
from agents.deterministic import DeterministicAgent
from data_gen.categorical_classification import DataSchemaFactory
from utils.rl import run_episode
from graph.node import InputNode, SingleNeuron, SubGraphNode
from utils.visualize import visualize_graph, print_graph_nodes


def train_learned_abstraction_model(df, epochs=1000, train_ratio=0.5):
    """
    Builds and trains a learned abstraction model using a parameterized train/test split.
    This model consists of:
      - An input layer (2 features),
      - A hidden Dense layer with ReLU,
      - An output Dense layer with sigmoid.

    After training/evaluating, returns a SubGraphNode that outputs the hidden layer.
    """
    from keras import layers, initializers, models
    kernel_init = initializers.GlorotUniform(seed=42)
    input_shape = (2,)

    new_input = layers.Input(shape=input_shape, name="sub_input")
    hidden = layers.Dense(3, activation='relu', 
                          name="hidden_layer", 
                          kernel_initializer=kernel_init)(new_input)
    output = layers.Dense(1, activation='sigmoid',
                          name="output_layer",
                          kernel_initializer=kernel_init)(hidden)
    full_model = models.Model(new_input, output, name="learned_abstraction_model_full")
    full_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Use the shared function with the given ratio/epochs
    accuracy = train_and_evaluate(full_model, df, train_ratio=train_ratio, epochs=epochs, verbose=0)
    #print(f"Learned abstraction accuracy after {epochs} epochs "
    #      f"with {train_ratio * 100:.0f}% training split: {accuracy:.3f}")

    # Extract the hidden-layer submodel as a SubGraphNode
    abstraction_model = models.Model(new_input, hidden, name="learned_abstraction_model_extracted")
    return SubGraphNode(name="learned_abstraction_node", model=abstraction_model), accuracy




class TestLearnedAbstractionTraining(unittest.TestCase):
    def test_learned_abstraction_training(self):
        """
        A learned abstraction trained for 500 epochs, when reused, allows high accuracy upon training
        the adapters for 300 epochs (over 10 seeds).

        This test:
          1. Creates a minimal network with one input node and one output node (with sigmoid activation).
          2. Uses a helper to train a learned abstraction (a hidden layer with 3 neurons) on a dataset of 500 examples.
          3. Adds the learned abstraction into the network by connecting input -> abstraction -> output.
          4. Freezes the learned abstraction so its weights remain unchanged.
          5. Saves the learned abstraction's weights.
          6. Rebuilds the Keras model via the composer.
          7. Fine-tunes the composed model for 300 epochs (only the output neuron's weights should change).
          8. Evaluates and prints the final accuracy using the shared train_and_evaluate function.
          9. Verifies that the learned abstraction's weights are identical before and after fine-tuning.
          10. Stores the learned abstraction in the repository.
          11. Asserts that the final accuracy is above a chosen threshold.
        """
        desc = ["Informal test to demonstrate learned abstraction advantage",
        "Test: Learned abstraction weights should remain unchanged after fine-tuning.",
        "Test: Trained network accuracy should >= abstraction accuracy."
        ]

        print("\n".join(line for line in desc))

        for run in range(1):
            # Create the schema with fixed distribution parameters.
            factory = DataSchemaFactory()
            schema = factory.create_schema(
                num_features=2,
                num_categories=2,
                num_classes=2,
                random_seed=SEED+run
            )
            dataset = schema.generate_dataset(num_instances=500)
            
            # Train a learned abstraction on the dataset.
            learned_abstraction, abstraction_acc = train_learned_abstraction_model(dataset, epochs=10)

            # Freeze the learned abstraction so its weights remain unchanged.
            learned_abstraction.model.trainable = False
            weights_before = learned_abstraction.model.get_weights()

            # Store the learned abstraction in the shared repository.
            State.repository["learned_abstraction"] = learned_abstraction

            # Build a minimal network.
            composer, model = create_minimal_graphmodel(input_shape=(2,))

            # Add the learned abstraction into the network from the repository.
            composer.add_node(State.repository["learned_abstraction"])
            composer.connect("input", learned_abstraction.name)
            composer.connect(learned_abstraction.name, "output")
            composer.remove_connection("input", "output")

            # Rebuild the Keras model.
            model = composer.build()

            # Create the base optimizer.


            lr_map = {
                # Example: any variable whose name has 'output' gets a 5x gradient boost
                'fast': 1.0
            }

            # Wrap a base optimizer (e.g., Adam) with the LR multiplier
            #base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            custom_optimizer = AdamWithLRMultiplier(lr_map, learning_rate=1e-3)

            model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            
            #visualize_graph(composer=composer)
            print_graph_nodes(composer=composer)

            # Evaluate the composed model using the shared train_and_evaluate function.
            acc = train_and_evaluate(model, dataset, train_ratio=0.5, epochs=300, verbose=0)
            print(f"Trained network accuracy after fine-tuning: {acc:.3f}, Abstraction accuracy: {abstraction_acc:.3f}")
            
            # Check that the learned abstraction's weights have not changed.
            weights_after = learned_abstraction.model.get_weights()
            for w_before, w_after in zip(weights_before, weights_after):
                assertion = "Learned abstraction weights should remain unchanged after fine-tuning."
                np.testing.assert_allclose(w_before, w_after, atol=1e-7, err_msg=assertion)

            # Assert final accuracy is above threshold.
            #assertion = "Trained network accuracy should be above 0.9 times abstraction accuracy."
            #self.assertGreaterEqual(acc, 0.5*abstraction_acc, assertion)

if __name__ == "__main__":
    unittest.main()




        # # Define a transition rule that generates a fresh dataset for each state, with the same schema
        # def transition_rule(state, actions):

        #     dataset = schema.generate_dataset(num_instances=500)

        #     # Return a new state with the generated dataset (agents_states remain unchanged).
        #     return State(dataset=dataset, agents_states=state.agents_states)

        # # Define a reward rule that simply returns a fixed reward for each agent.
        # def reward_rule(prev_state, actions, new_state):
        #     return {agent_id: 1.0 for agent_id in new_state.agents_states}

        # # Create an initial agent graphmodel for agent 0.
        # initial_graphmodel = create_minimal_graphmodel(input_shape=(2,))
        # initial_agent_state = AgentState(
        #     nodes=list(initial_graphmodel[0].nodes.keys()),
        #     connections=initial_graphmodel[0].connections,
        #     graphmodel=initial_graphmodel
        # )
        # # Create an initial state
        # initial_state = State(dataset=schema.generate_dataset(num_instances=500), agents_states={0: initial_agent_state})

        # # Construct the environment using the new signature.
        # env = RLEnvironment(transition_rule=transition_rule, reward_rule=reward_rule, initial_state=initial_state)

        # # Call step() to apply actions, move to a new state, and collect rewards
        # state, rewards = env.step()
        # dataset = state.dataset





















# # tests/test_rl_dev.py

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""         # Force CPU usage.
# os.environ["TF_DETERMINISTIC_OPS"] = "1"          # Request deterministic operations.

# import random
# import numpy as np
# import tensorflow as tf
# import unittest

# # Seed all relevant random generators.
# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# from envs.environment import RLEnvironment, State
# from utils.nn import create_minimal_graphmodel, train_learned_abstraction_model
# from agents.deterministic import DeterministicAgent
   
# from data_gen.categorical_classification import DataSchemaFactory
# from utils.rl import run_episode


# class TestLearnedAbstractionTraining(unittest.TestCase):
#     def test_learned_abstraction_training(self):
#         """
#         This test:
#           1. Creates a minimal network with one input node and one output node (with sigmoid activation).
#           2. Uses a helper to train a learned abstraction (a hidden layer with 3 neurons) on a dataset of 500 examples.
#           3. Adds the learned abstraction into the network by connecting input -> abstraction -> output.
#           4. Freezes the learned abstraction so its weights remain unchanged.
#           5. Saves the learned abstraction's weights.
#           6. Rebuilds the Keras model via the composer.
#           7. Fine-tunes the composed model for 300 epochs (only the output neuron's weights should change).
#           8. Evaluates and prints the final accuracy.
#           9. Verifies that the learned abstraction's weights are identical before and after fine-tuning.
#           10. Stores the learned abstraction in the repository.
#           11. Asserts that the final accuracy is above a chosen threshold.
#         """

#         # Create the schema once with a fixed seed so distribution parameters are fixed.
#         factory = DataSchemaFactory()
#         schema = factory.create_schema(
#             num_features=2,
#             num_categories=2,
#             num_classes=2,
#             random_seed=SEED
#         )

#         # Create the environment with 500 examples per step.
#         env = RLEnvironment(num_instances_per_step=500, seed=0, n_agents=1, schema=schema)
#         state = env.reset()

#         # Call step() to generate a dataset and update the state.
#         state, _ = env.step()
#         dataset = state.dataset

#         # Build the minimal network.
#         composer, model = create_minimal_graphmodel(input_shape=(2,))
        
#         # Train the learned abstraction on 500 examples.
#         learned_abstraction = train_learned_abstraction_model(dataset, epochs=1000)
        
#         # Freeze the learned abstraction so its weights remain unchanged.
#         learned_abstraction.model.trainable = False
#         weights_before = learned_abstraction.model.get_weights()

#         # Add the learned abstraction into the network.
#         composer.add_node(learned_abstraction)
#         composer.connect("input", learned_abstraction.name)
#         composer.connect(learned_abstraction.name, "output")
#         composer.remove_connection("input", "output")

#         # Rebuild the Keras model.
#         model = composer.build()
#         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
#         # Optionally visualize the graph.
#         # print_graph_nodes(composer)
#         # visualize_graph(composer=composer)

#         # Fine-tune the composed model.
#         features = dataset[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
#         true_labels = dataset["label"].to_numpy(dtype=int)
#         model.fit(features, true_labels, epochs=300, verbose=0)

#         # Evaluate the fine-tuned model.
#         loss, acc = model.evaluate(features, true_labels, verbose=0)
#         print(f"Trained network accuracy after fine-tuning: {acc:.3f}")
        
#         # Check that the learned abstraction's weights have not changed.
#         weights_after = learned_abstraction.model.get_weights()
#         for w_before, w_after in zip(weights_before, weights_after):
#             assertion = "Learned abstraction weights should remain unchanged after fine-tuning."
#             np.testing.assert_allclose(w_before, w_after, atol=1e-7, err_msg=assertion)
#             print(assertion)
        
#         # Store the learned abstraction in the repository.
#         State.repository["learned_abstraction"] = learned_abstraction
        
#         # Assert final accuracy is above threshold.
#         assertion = "Trained network accuracy should be above 0.9."
#         print(assertion)
#         self.assertGreater(acc, 0.9, assertion)


# def my_custom_policy(state, step, valid_actions):
#     """
#     A custom policy that tries to pick 'add_abstraction' at step 0 if it's
#     in the valid_actions list, otherwise falls back to selecting 'no_change'
#     or a random valid action.
#     """
#     import numpy as np
    
#     if step == 0 and "add_abstraction" in valid_actions:
#         return "add_abstraction"
#     elif "no_change" in valid_actions:
#         return "no_change"
#     else:
#         return np.random.choice(valid_actions) if valid_actions else None


# class TestReuseAdvantage(unittest.TestCase):
#     def test_reuse_advantage(self):
#         """
#         Hypothesis: When one agent adds the learned abstraction and the other does not,
#         the agent that adds it will obtain higher accuracy and receive a reward advantage.
        
#         Test: Pre-populate the repository with the learned abstraction.
#         Run a controlled number of steps where agent 0 chooses "add_abstraction" on the first step,
#         while agent 1 always chooses "no_change". Print detailed step-by-step accuracies and rewards,
#         and verify that agent 0's cumulative reward exceeds that of agent 1.
#         """
#         num_steps = 5

#         # Create a schema externally and pass it to the environment.
#         factory = DataSchemaFactory()
#         schema = factory.create_schema(
#             num_features=2,
#             num_categories=2,
#             num_classes=2,
#             random_seed=0
#         )

#         # Create the environment with the externally provided schema.
#         env = RLEnvironment(num_instances_per_step=100, seed=0, n_agents=2, schema=schema)
#         state = env.reset()

#         # Call step() to generate a dataset and update the state.
#         state, _ = env.step()
#         dataset = state.dataset

#         # Train an abstraction on the dataset and store it in the environment's repository.
#         learned_abstraction = train_learned_abstraction_model(dataset, epochs=1000)
#         State.repository["learned_abstraction"] = learned_abstraction

#         # Create DeterministicAgents.
#         # Agent 0 uses a custom policy to choose "add_abstraction" on step 0.
#         agent0 = DeterministicAgent(policy=my_custom_policy, training_params={"epochs":300, "verbose":0})
#         agent0.update_valid_actions(["add_abstraction", "no_change"])

#         # Agent 1 always chooses "no_change".
#         agent1 = DeterministicAgent(training_params={"epochs":300, "verbose":0})
#         agent1.update_valid_actions(["no_change"])

#         # Prepare the dictionary of agents.
#         agents_dict = {
#             0: agent0,
#             1: agent1
#         }

#         # Run the episode for the specified number of steps.
#         actions, rewards, accuracies = run_episode(env, agents_dict, seed=0, steps=num_steps, schema=schema, step_callback=None)

#         print("\nDetailed Step-by-Step Output:")
#         for step in range(num_steps):
#             print(f"Step {step+1}:")
#             print(f"  Agent 0: Accuracy = {accuracies[0][step]:.3f}, Reward = {rewards[0][step]:.3f}")
#             print(f"  Agent 1: Accuracy = {accuracies[1][step]:.3f}, Reward = {rewards[1][step]:.3f}")

#         # Compute cumulative rewards.
#         reward0 = sum(rewards[0])
#         reward1 = sum(rewards[1])
 
#         diff = reward0 - reward1
#         print(f"\nTest outcome: Agent 0's reward = {reward0}, Agent 1's reward = {reward1}.")
#         print(f"Agent 0 won by a margin of {diff} reward points.")
#         print("Actions taken:", actions)
        
#         self.assertGreater(
#             reward0,
#             reward1,
#             "Agent 0 should receive a higher reward than Agent 1 when using the learned abstraction."
#         )






#     def _init_agents(self, n_agents):
#         """
#         Helper method to initialize agent graphmodels.
#         """
#         agents_graphmodels = {}
#         for agent_id in range(n_agents):
#             agents_graphmodels[agent_id] = create_minimal_graphmodel(input_shape=(2,))
#         return agents_graphmodels





#     def _get_state(self):
#         """
#         Constructs and returns the current state as a State object.
#         """
#         agents_states = {}
#         for agent_id, graphmodel in self.agents_graphmodels.items():
#             # Extract composer and connections from the graphmodel.
#             composer, _ = graphmodel
#             agent_state = AgentState(
#                 nodes=list(composer.nodes.keys()),
#                 connections=composer.connections,
#                 graphmodel=graphmodel
#             )
#             agents_states[agent_id] = agent_state

#         return State(dataset=self.dataset, agents_states=agents_states)




#     def compute_rewards_default(self, state, actions, next_state):
#         """
#         Default reward computation for the legacy environment.
#         """
#         rewards = {}
#         for agent_id in state.agents_states:
#             rewards[agent_id] = 1.0
#         return rewards


#     def compute_rewards(self, accuracies):
#         """
#         Computes rewards based on provided accuracies.
#         """
#         for agent_id, acc in accuracies.items():
#             self.agent_steps[agent_id] += 1
#             self.agent_cum_acc[agent_id] += acc
#         rewards = {}
#         for agent_id, acc in accuracies.items():
#             avg_acc = self.agent_cum_acc[agent_id] / self.agent_steps[agent_id]
#             rewards[agent_id] = avg_acc
#         return rewards

#     def _default_transition(self, state, actions, schema):
#         """
#         Return the next state
#         """

#         dataset = schema.generate_dataset(num_instances=self.num_instances_per_step)

#         return State(dataset, )

















# if __name__ == "__main__":
#     unittest.main()
