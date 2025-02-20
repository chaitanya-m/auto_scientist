import copy
import unittest
import numpy as np
from utils.environment import RLEnvironment, run_episode, create_minimal_network, train_learned_abstraction_model
from agents.deterministic import DeterministicAgent
from agents.qlearning import QLearningAgent
from data_gen.categorical_classification import DataSchemaFactory

class TestRLAgentCatchUp(unittest.TestCase):
    def test_deterministic_reuse_catchup_to_baseline(self):
        """
        Agent 0 always reuses a learned abstraction. It should be able to match the pretrained accuracy, 
        which on this dataset is always 1.

        1 step per episode.

        As the problem is simple enough that the learned abstraction reaches accuracy 1.0 on all datasets,
        the initial weights' randomness shouldn't factor into the ability of the agents to learn, given
        sufficient epochs.

        We want the number of learning epochs for reuse to be lower.
        """

        # 1. Create a schema externally and pass it to the environment.
        factory = DataSchemaFactory()
        schema = factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=0
        )
        num_steps = 1
        env = RLEnvironment(total_steps=num_steps, num_instances_per_step=300, seed=0, n_agents=1, schema=schema)
   
        agent0 = DeterministicAgent()
        agent0.update_valid_actions(["add_abstraction"])

        n_episodes = 10
        for ep in range(n_episodes):
            # Reset environment each episode for a fresh start.
            env.reset(seed=ep)
            env.step()  # Generate dataset
            dataset = env._get_state().dataset

            # Train an abstraction on the new dataset, store it in the repository, replacing any existing model.
            learned_abstraction = train_learned_abstraction_model(dataset, epochs=500)
            env.repository["learned_abstraction"] = learned_abstraction

            # Reset the agent's network so that previous episodes do not interfere.
            env.agents_networks[0] = create_minimal_network(input_shape=(2,))
            
            agents_dict = {0: agent0}

            actions_history, rewards, accuracies = run_episode(env, agents_dict, seed=ep)

            # Print stepwise and episode-wise accuracies for the agent.
            for agent_id in sorted(accuracies.keys()):
                print(f"Agent {agent_id} stepwise accuracies: {accuracies[agent_id]}")
                print(actions_history[agent_id])
                #avg_acc = np.mean(accuracies[agent_id])
                #print(f"Agent {agent_id} average accuracy for episode {ep}: {avg_acc:.4f}")

        final_acc0 = np.mean(accuracies[0])  # average accuracy for agent0
        self.assertGreaterEqual(
            final_acc0, 0.9, 
            "Agent 0 failed to catch up to at least 90% of ideal performance."
        )

if __name__ == '__main__':
    unittest.main()




    # def test_agent1_catches_up_with_pretrained_abstraction(self):
    #     """
    #     Agent 0 uses a strong fixed policy. Agent 1, a Q-learning agent, can 
    #     optionally insert the pretrained learned abstraction from env.repository.
    #     Over multiple steps or episodes, agent 1 should learn to add the abstraction 
    #     at some point, matching or beating agent 0's performance.
    #     """

    #     # 1. Create environment
    #     num_steps = 1
    #     env = RLEnvironment(total_steps=num_steps, num_instances_per_step=300, seed=0)
   
    #     agent0 = DeterministicAgent()
    #     agent0.update_valid_actions(["add_abstraction"])

    #     # 4. Agent1 (Q-learning) can choose from ["add_abstraction", "no_change"].
    #     #    We'll let it discover when "add_abstraction" is beneficial.
    #     agent1 = QLearningAgent(action_space=["no_change", "add_abstraction"], epsilon=0.5, epsilon_decay=0.5)
    #     agent1.update_valid_actions(["no_change", "add_abstraction"])

    #     # 5. Possibly run multiple episodes so agent1 can learn. For a single test, 
    #     #    we can do a single episode or a small batch:
    #     n_episodes = 10
    #     for ep in range(n_episodes):
    #         # Reset environment each episode for a fresh start.
    #         env.reset(seed=ep)

    #         # Train an abstraction on new dataset, store it in the repository
    #         learned_abstraction = train_learned_abstraction_model(env, epochs=500)
    #         env.repository["learned_abstraction"] = learned_abstraction

    #         # Reset each agent's model so that previous episodes do not interfere.

    #         env.agents_networks[0] = create_minimal_network(input_shape=(2,))
    #         env.agents_networks[1] = create_minimal_network(input_shape=(2,))
            
    #         agents_dict = {0: agent0, 1: agent1}

    #         actions_history, rewards, accuracies = run_episode(env, agents_dict, seed=ep)
    #         # (Additional Q-learning updates could be performed here, if needed.)

    #         # Print stepwise and episode-wise accuracies for each agent.
    #         for agent_id in sorted(accuracies.keys()):
    #             print(f"Agent {agent_id} stepwise accuracies: {accuracies[agent_id]}")
    #             print(actions_history[agent_id])
    #             #avg_acc = np.mean(accuracies[agent_id])
    #             #print(f"Agent {agent_id} average accuracy for episode {ep}: {avg_acc:.4f}")


    #     # 6. Final check: agent1's performance is near or above agent0's in the last episode

    #     final_acc0 = np.mean(accuracies[0])  # average accuracy for agent0
    #     final_acc1 = np.mean(accuracies[1])  # average accuracy for agent1
    #     self.assertGreaterEqual(
    #         final_acc1, final_acc0 * 0.9, 
    #         "Agent 1 failed to catch up to at least 90% of agent 0's performance."
    #     )

if __name__ == "__main__":
    unittest.main()