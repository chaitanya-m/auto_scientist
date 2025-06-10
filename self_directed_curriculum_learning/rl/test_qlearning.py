"""test_qlearning.py

Unit tests to verify eligibility trace behavior and ensure traces work as intended.
Tests cover trace initialization, decay, updates, and edge cases.
"""

import unittest
from collections import defaultdict
from typing import List
import numpy as np
import random  # Add this import

# Import the classes we want to test
from qlearning import TabularQ, EligibilityTraceCriticUpdater
from interfaces import Transition


class TestEligibilityTraces(unittest.TestCase):
    """Test suite for eligibility trace functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.q_table = TabularQ(alpha=0.1, gamma=0.9, init=0.0)
        self.alpha = 0.1
        self.gamma = 0.9
        self.lambda_ = 0.8
        
        self.updater = EligibilityTraceCriticUpdater(
            q_table=self.q_table,
            alpha=self.alpha,
            gamma=self.gamma,
            lambda_=self.lambda_
        )
    
    def create_transition(self, state: int, action: int, reward: float, 
                         next_state: int, done: bool = False) -> Transition:
        """Helper to create transitions for testing."""
        return Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={}
        )
    
    def test_trace_initialization(self):
        """Test that traces start empty and get initialized correctly."""
        # Initially no traces
        self.assertEqual(len(self.updater.traces), 0)
        
        # After processing one transition, should have one trace
        transition = self.create_transition(state=0, action=1, reward=5.0, next_state=1)
        self.updater.step([transition])
        
        self.assertEqual(len(self.updater.traces), 1)
        self.assertEqual(self.updater.traces[0][1], 1.0)  # Trace should be 1.0
    
    def test_replacing_traces(self):
        """Test that traces are replaced (not accumulated) when state-action is revisited."""
        # Visit state 0, action 1 twice
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=1.0, next_state=0),  # Different state
            self.create_transition(state=0, action=1, reward=1.0, next_state=2),  # Revisit same state-action
        ]
        
        self.updater.step(transitions)
        
        # The trace for (0,1) should be 1.0 (replaced), not accumulated
        self.assertEqual(self.updater.traces[0][1], 1.0)
    
    def test_trace_decay(self):
        """Test that traces decay correctly over time."""
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=1.0, next_state=2),
        ]
        
        self.updater.step(transitions)
        
        # After first transition: trace[0][1] = 1.0
        # After second transition: trace[0][1] should decay by gamma * lambda
        expected_decay = self.gamma * self.lambda_  # 0.9 * 0.8 = 0.72
        self.assertAlmostEqual(self.updater.traces[0][1], expected_decay, places=5)
        self.assertEqual(self.updater.traces[1][0], 1.0)  # New trace should be 1.0
    
    def test_trace_cleanup(self):
        """Test that small traces are removed for efficiency."""
        # Create a transition that will generate a very small trace after decay
        transitions = []
        
        # Create many transitions to force trace decay below threshold
        for i in range(20):  # Many steps to force decay below 0.01
            transitions.append(
                self.create_transition(state=i, action=0, reward=1.0, next_state=i+1)
            )
        
        self.updater.step(transitions)
        
        # Early traces should be cleaned up (< 0.01)
        # Check that state 0's trace was removed
        self.assertNotIn(0, self.updater.traces)
        
        # But recent traces should still exist
        self.assertIn(19, self.updater.traces)  # Last state should have trace
    
    def test_episode_boundary_clears_traces(self):
        """Test that traces are cleared when episode ends."""
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=10.0, next_state=2, done=True),
        ]
        
        self.updater.step(transitions)
        
        # After episode ends, traces should be cleared
        self.assertEqual(len(self.updater.traces), 0)
    
    def test_td_error_propagation(self):
        """Test that TD errors propagate to all traced state-actions."""
        # Set up initial Q-values to create predictable TD error
        self.q_table._table[0][1] = 5.0  # Q(s0, a1) = 5.0
        self.q_table._table[1][0] = 3.0  # Q(s1, a0) = 3.0
        self.q_table._table[2][1] = 0.0  # Q(s2, a1) = 0.0 (default)
        
        transitions = [
            self.create_transition(state=0, action=1, reward=2.0, next_state=1),
            self.create_transition(state=1, action=0, reward=8.0, next_state=2, done=True),
        ]
        
        # Store initial Q-values
        initial_q_0_1 = self.q_table.q(0, 1)
        initial_q_1_0 = self.q_table.q(1, 0)
        
        self.updater.step(transitions)
        
        # Both state-actions should have been updated
        final_q_0_1 = self.q_table.q(0, 1)
        final_q_1_0 = self.q_table.q(1, 0)
        
        # Values should have changed due to updates
        self.assertNotEqual(initial_q_0_1, final_q_0_1)
        self.assertNotEqual(initial_q_1_0, final_q_1_0)
    
    def test_lambda_zero_equals_td(self):
        """Test that λ=0 behaves like standard TD (only current state updated)."""
        # Create updater with λ=0
        td_updater = EligibilityTraceCriticUpdater(
            q_table=TabularQ(alpha=0.1, gamma=0.9),
            alpha=0.1,
            gamma=0.9,
            lambda_=0.0
        )
        
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=1.0, next_state=2),
        ]
        
        td_updater.step(transitions)
        
        # With λ=0, after first transition, trace for (0,1) should decay to 0
        # After second transition, only (1,0) should have non-zero trace
        self.assertLess(td_updater.traces[0][1], 0.01)  # Should be cleaned up
        self.assertEqual(td_updater.traces[1][0], 1.0)
    
    def test_lambda_one_behavior(self):
        """Test that λ=1 maintains traces throughout episode."""
        # Create updater with λ=1
        mc_updater = EligibilityTraceCriticUpdater(
            q_table=TabularQ(alpha=0.1, gamma=0.9),
            alpha=0.1,
            gamma=0.9,
            lambda_=1.0
        )
        
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=1.0, next_state=2),
            self.create_transition(state=2, action=1, reward=1.0, next_state=3),
        ]
        
        mc_updater.step(transitions)
        
        # With λ=1, traces should persist (decay by γ only)
        expected_trace_0 = (0.9) ** 2  # Decayed by γ twice
        expected_trace_1 = 0.9         # Decayed by γ once
        expected_trace_2 = 1.0         # Current trace
        
        self.assertAlmostEqual(mc_updater.traces[0][1], expected_trace_0, places=5)
        self.assertAlmostEqual(mc_updater.traces[1][0], expected_trace_1, places=5)
        self.assertEqual(mc_updater.traces[2][1], expected_trace_2)
    
    def test_step_returns_metrics(self):
        """Test that step method returns useful metrics."""
        transitions = [
            self.create_transition(state=0, action=1, reward=1.0, next_state=1),
            self.create_transition(state=1, action=0, reward=5.0, next_state=2),
        ]
        
        metrics = self.updater.step(transitions)
        
        # Should return metrics dictionary
        self.assertIsInstance(metrics, dict)
        self.assertIn("td_error", metrics)
        self.assertIn("active_traces", metrics)
        self.assertIn("trace_updates", metrics)
        
        # Metrics should be reasonable
        self.assertGreaterEqual(metrics["td_error"], 0.0)
        self.assertGreaterEqual(metrics["active_traces"], 0)
        self.assertGreaterEqual(metrics["trace_updates"], 0)
    
    def test_empty_experience_handling(self):
        """Test that empty experience lists are handled gracefully."""
        metrics = self.updater.step([])
        
        # Should not crash and return reasonable metrics
        self.assertEqual(metrics["td_error"], 0.0)
        self.assertEqual(metrics["active_traces"], 0)
        self.assertEqual(metrics["trace_updates"], 0)
    
    def test_trace_values_reasonable(self):
        """Test that trace values stay within reasonable bounds."""
        transitions = []
        
        # Create a longer episode
        for i in range(10):
            transitions.append(
                self.create_transition(state=i, action=0, reward=1.0, next_state=i+1)
            )
        
        self.updater.step(transitions)
        
        # All traces should be between 0 and 1
        for state_traces in self.updater.traces.values():
            for trace_value in state_traces.values():
                self.assertGreaterEqual(trace_value, 0.0)
                self.assertLessEqual(trace_value, 1.0)


class TestQLearning(unittest.TestCase):
    """Test suite for Q-learning agent functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import gym here to avoid issues if not installed
        try:
            import gym
            self.gym = gym
        except ImportError:
            self.skipTest("gym not available")
    
    def test_qlearning_cartpole_learning_trend(self):
        """Test that Q-learning shows upward trend in moving average performance."""
        from qlearning import create_qlearning_agent
        
        # Test parameters - simplified for trend detection
        seed = 42  # Single seed is fine for trend testing
        episodes = 300
        window_size = 50  # Moving average window
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Create environment and agent
        env = self.gym.make("CartPole-v1")
        agent = create_qlearning_agent(
            env, 
            alpha=0.2,  # Higher learning rate for clearer trends
            gamma=0.99, 
            epsilon=0.2,  # Higher exploration
            bins=8  # Simpler discretization
        )
        
        # Collect all episode rewards
        episode_rewards = []
        
        for episode in range(episodes):
            # Learn for one episode
            reward = agent.learn(max_steps=env._max_episode_steps, update_frequency=-1)
            episode_rewards.append(reward)
        
        env.close()
        
        # Calculate moving averages
        moving_averages = []
        for i in range(window_size, len(episode_rewards)):
            window = episode_rewards[i-window_size:i]
            moving_avg = np.mean(window)
            moving_averages.append(moving_avg)
        
        # Test 1: Moving average should show upward trend
        # Compare first quarter vs last quarter of moving averages
        quarter_size = len(moving_averages) // 4
        early_avg = np.mean(moving_averages[:quarter_size])
        late_avg = np.mean(moving_averages[-quarter_size:])
        
        improvement = late_avg - early_avg
        
        self.assertGreater(improvement, 10.0,
            f"Moving average should show upward trend. "
            f"Early: {early_avg:.1f}, Late: {late_avg:.1f}, "
            f"Improvement: {improvement:.1f}")
        
        # Test 2: Trend direction - fit a line to moving averages
        x = np.arange(len(moving_averages))
        slope, intercept = np.polyfit(x, moving_averages, 1)
        
        self.assertGreater(slope, 0.01,
            f"Moving average trend should be positive. Slope: {slope:.4f}")
        
        # Test 3: Final moving average should be better than initial
        final_moving_avg = moving_averages[-1]
        initial_moving_avg = moving_averages[0]
        
        self.assertGreater(final_moving_avg, initial_moving_avg,
            f"Final moving average ({final_moving_avg:.1f}) should exceed "
            f"initial moving average ({initial_moving_avg:.1f})")
        
        # Test 4: Majority of moving average should show improvement
        # Count how many moving averages are better than the first one
        improvements = np.array(moving_averages) > initial_moving_avg
        improvement_ratio = np.mean(improvements)
        
        self.assertGreater(improvement_ratio, 0.6,
            f"Most of training should show improvement over initial performance. "
            f"Improvement ratio: {improvement_ratio:.2f}")
    
    def test_qlearning_basic_functionality(self):
        """Test basic Q-learning functionality on a simple environment."""
        from qlearning import create_qlearning_agent
        
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        env = self.gym.make("CartPole-v1")
        
        agent = create_qlearning_agent(
            env,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1
        )
        
        # Test that agent can learn without crashing
        initial_q_values = dict(agent.critic_updater.q_table._table)
        
        # Run a few episodes
        rewards = []
        for _ in range(10):
            reward = agent.learn(max_steps=env._max_episode_steps, update_frequency=-1)
            rewards.append(reward)
        
        # Test 1: Agent should produce reasonable rewards
        self.assertTrue(all(r >= 0 for r in rewards), "All rewards should be non-negative")
        self.assertTrue(any(r > 10 for r in rewards), "Should achieve some success")
        
        # Test 2: Q-table should have been updated
        final_q_values = dict(agent.critic_updater.q_table._table)
        self.assertNotEqual(initial_q_values, final_q_values, "Q-table should be updated")
        
        # Test 3: Q-values should be reasonable
        all_q_values = []
        for state_dict in final_q_values.values():
            all_q_values.extend(state_dict.values())
        
        if all_q_values:  # If we have any Q-values
            max_q = max(all_q_values)
            min_q = min(all_q_values)
            
            # Q-values should be reasonable for CartPole (not extremely large/small)
            self.assertLess(abs(max_q), 1000, f"Max Q-value seems too large: {max_q}")
            self.assertLess(abs(min_q), 1000, f"Min Q-value seems too large: {min_q}")
        
        env.close()
    
    def test_qlearning_deterministic_behavior(self):
        """Test that Q-learning behaves deterministically with same seed."""
        from qlearning import create_qlearning_agent
        
        def run_qlearning_episode(seed):
            """Helper to run one episode with given seed."""
            # Set ALL random seeds
            random.seed(seed)
            np.random.seed(seed)
            
            env = self.gym.make("CartPole-v1")
            
            agent = create_qlearning_agent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
            
            # Reset environment with seed for deterministic initial state
            obs = env.reset(seed=seed)
            if isinstance(obs, tuple):
                obs = obs[0]
            
            # Manually run one episode to control randomness better
            total_reward = 0.0
            done = False
            steps = 0
            max_steps = env._max_episode_steps or 500
            
            while not done and steps < max_steps:
                # Get action from policy
                action = agent.target_policy.action(obs, deterministic=False)  # Keep stochastic for learning
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    next_obs, reward, term, trunc, _ = step_result
                    done = term or trunc
                else:
                    next_obs, reward, done, _ = step_result
                
                total_reward += reward
                
                # Create transition for learning
                from interfaces import Transition
                transition = Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    info={}
                )
                
                # Update critic with single transition
                agent.critic_updater.step([transition])
                
                obs = next_obs
                steps += 1
            
            # Get some Q-values for comparison
            q_sample = []
            for state_dict in list(agent.critic_updater.q_table._table.values())[:3]:
                q_sample.extend(list(state_dict.values())[:2])
            
            env.close()
            return total_reward, q_sample
        
        # Run same seed twice
        seed = 42
        reward1, q_values1 = run_qlearning_episode(seed)
        reward2, q_values2 = run_qlearning_episode(seed)
        
        # Should get identical results
        self.assertEqual(reward1, reward2, "Same seed should give same reward")
        self.assertEqual(len(q_values1), len(q_values2), "Should have same number of Q-values")
        
        for q1, q2 in zip(q_values1, q_values2):
            self.assertAlmostEqual(q1, q2, places=10, 
                msg="Same seed should give identical Q-values")


if __name__ == "__main__":
    unittest.main()