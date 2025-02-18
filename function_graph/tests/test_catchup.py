#tests/test_catchup.py
import unittest
import numpy as np
from utils.rl_dev import RLEnvironment, run_episode
from tests.test_rldev import train_learned_abstraction_model
from utils.rl_dev import create_minimal_network, DummyAgent
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
        num_steps = 10
        env = RLEnvironment(total_steps=num_steps, num_instances_per_step=100, seed=42)
        
        # 2. Train or load an abstraction, store it in the repository
        learned_abstraction = train_learned_abstraction_model(env, epochs=1000)
        env.repository["learned_abstraction"] = learned_abstraction

        # 3. Agent0 (fixed policy) always chooses "no_change" (or some known good sequence).
        action_plan0 = ["add_abstraction"] + ["no_change"] * (num_steps-1)
        agent0 = DummyAgent(action_plan={0: action_plan0})

        # 4. Agent1 (Q-learning) can choose from ["add_abstraction", "no_change"].
        #    We'll let it discover when "add_abstraction" is beneficial.
        agent1 = QLearningAgent(action_space=["no_change", "add_abstraction"])

        # 5. Possibly run multiple episodes so agent1 can learn. For a single test, 
        #    we can do a single episode or a small batch:
        n_episodes = 5
        for ep in range(n_episodes):
            # Reset environment each episode if you want fresh starts
            # or keep it continuous if you prefer online learning
            agents_dict = {0: agent0, 1: agent1}
            actions, rewards, accuracies = run_episode(env, agents_dict, seed=42)
            
            # After each step, agent1 sees the reward and can update Q
            # So you'd modify run_episode or handle after run_episode in a loop
            # to call agent1.update_q(...) for each step or something similar.

            # Example: if run_episode doesn't do per-step Q-learning updates,
            # we might want to modify the environment or run_episode to pass
            # next_state and done flags each iteration.

        # 6. Final check: agent1's performance is near or above agent0's in the last episode
        final_acc0 = np.mean(accuracies[0])  # average accuracy for agent0
        final_acc1 = np.mean(accuracies[1])  # average accuracy for agent1
        self.assertGreaterEqual(
            final_acc1, final_acc0 * 0.9, 
            "Agent 1 failed to catch up to at least 90% of agent 0's performance."
        )
