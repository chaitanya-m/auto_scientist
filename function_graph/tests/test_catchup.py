#tests/test_catchup.py
import unittest
import numpy as np
from utils.environment import RLEnvironment, run_episode, create_minimal_network
from tests.test_rldev import train_learned_abstraction_model
from agents.deterministic import DeterministicAgent
from agents.qlearning import QLearningAgent

class TestRLAgentCatchUp(unittest.TestCase):
    def test_agent1_catches_up_with_pretrained_abstraction(self):
        """
        Agent 0 uses a strong fixed policy. Agent 1, a Q-learning agent, can 
        optionally insert the pretrained learned abstraction from env.repository.
        Over multiple steps or episodes, agent 1 should learn to add the abstraction 
        at some point, matching or beating agent 0's performance.
        """

        # 1. Create environment
        num_steps = 1
        env = RLEnvironment(total_steps=num_steps, num_instances_per_step=500, seed=0)

        # 3. Agent0 (fixed policy) always chooses "no_change" (or some known good sequence).
        fixed_action_plan = ["add_abstraction"] + ["no_change"] * (num_steps-1)
        agent0 = DeterministicAgent(action_plan=fixed_action_plan)

        # 4. Agent1 (Q-learning) can choose from ["add_abstraction", "no_change"].
        #    We'll let it discover when "add_abstraction" is beneficial.
        agent1 = QLearningAgent(action_space=["no_change", "add_abstraction"], epsilon=0.5, epsilon_decay=0.5)

        # 5. Possibly run multiple episodes so agent1 can learn. For a single test, 
        #    we can do a single episode or a small batch:
        n_episodes = 5
        for ep in range(n_episodes):
            # Reset environment each episode for a fresh start.
            env.reset(seed=ep)

            # Train an abstraction on new dataset, store it in the repository
            learned_abstraction = train_learned_abstraction_model(env, epochs=1000)
            env.repository["learned_abstraction"] = learned_abstraction

            # Reset each agent's model so that previous episodes do not interfere.

            env.agents_networks[0] = create_minimal_network(input_shape=(2,))
            env.agents_networks[1] = create_minimal_network(input_shape=(2,))
            
            agents_dict = {0: agent0, 1: agent1}

            # Reset agent 0's fixed action plan
            agent0.action_plan = fixed_action_plan

            actions, rewards, accuracies = run_episode(env, agents_dict, seed=ep)
            # (Additional Q-learning updates could be performed here, if needed.)


        # 6. Final check: agent1's performance is near or above agent0's in the last episode

        final_acc0 = np.mean(accuracies[0])  # average accuracy for agent0
        final_acc1 = np.mean(accuracies[1])  # average accuracy for agent1
        self.assertGreaterEqual(
            final_acc1, final_acc0 * 0.9, 
            "Agent 1 failed to catch up to at least 90% of agent 0's performance."
        )
