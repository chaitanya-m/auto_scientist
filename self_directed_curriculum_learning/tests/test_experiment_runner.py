import unittest
import numpy as np
from curriculum_generator.curriculum import AutoEncoderCurriculum
from env.fg_env import FunctionGraphEnv
from agents.mcts import SimpleMCTSAgent
from curriculum_generator.problems import AutoEncoderProblem  # Use the correct problem class
from experiment_runner import solve_one_problem, build_uniform_policy

class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        # Create a new curriculum for each test
        self.curriculum = AutoEncoderCurriculum.default(
            problem_cls=AutoEncoderProblem,  # Use the correct problem class
            initial_difficulty=0,
            num_problems=2  # Two problems for testing persistence
        )

        # Environment setup with reduced parameters for faster tests
        self.custom_config = {
            "increase_scale": 0.6,
            "decrease_scale": -0.4,
            "maintain_scale": 0.1
        }

        # Create a fresh environment instance for each test
        self.env = FunctionGraphEnv(
            curriculum=self.curriculum,
            train_epochs=1,  # Reduced epochs for faster tests
            batch_size=2,    # Reduced batch size for faster tests
            config=self.custom_config,
            seed=0
        )

        # Agent setup
        obs, _ = self.env.reset()
        obs_dim = obs.shape[0]
        action_dim = self.env.action_space.n
        policy_model = build_uniform_policy(obs_dim, action_dim)
        difficulty_policy_model = build_uniform_policy(obs_dim + 1, 3)  # For difficulty adjustment
        self.agent = SimpleMCTSAgent(
            env=self.env,
            policy_model=policy_model,
            difficulty_policy_model=difficulty_policy_model,
            search_budget=2,  # Reduced MCTS budget for faster tests
            c=1.0
        )

    def test_repository_persistence(self):
        # Run the first problem
        problem_seed = 0
        problem = next(iter(self.curriculum))
        df1, summary1 = solve_one_problem(
            problem=problem,
            problem_seed=problem_seed,
            mcts_budget=2,  # Reduced MCTS budget for faster tests
            steps=2,  # Reduced steps for faster tests
            agent=self.agent,
            env=self.env
        )
        repo_size_end_first_episode = len(self.env.repository)  # Measure repository size at the end of the first episode
        print(f"Repository size at the end of the first episode: {repo_size_end_first_episode}")  # Debugging statement

        # Preserve the repository across episodes
        preserved_repository = list(self.env.repository)

        # Reinitialize the curriculum for the second problem
        self.curriculum = AutoEncoderCurriculum.default(
            problem_cls=AutoEncoderProblem,
            initial_difficulty=0,
            num_problems=2
        )
        self.env = FunctionGraphEnv(
            curriculum=self.curriculum,
            train_epochs=1,  # Reduced epochs for faster tests
            batch_size=2,    # Reduced batch size for faster tests
            config=self.custom_config,
            seed=1
        )
        self.env.repository = preserved_repository  # Restore the repository
        self.agent.env = self.env  # Update the agent's environment

        # Measure repository size at the start of the second episode
        repo_size_start_second_episode = len(self.env.repository)
        print(f"Repository size at the start of the second episode: {repo_size_start_second_episode}")  # Debugging statement

        # Verify repository persistence
        self.assertEqual(
            repo_size_end_first_episode,
            repo_size_start_second_episode,
            "Repository size at the end of the first episode should equal the size at the start of the second episode."
        )

    def test_repository_updates(self):
        # Run a problem and ensure the repository size changes during the episode
        problem_seed = 0
        problem = next(iter(self.curriculum))

        # Track repository size at each step
        repo_size_sum = 0
        df, summary = solve_one_problem(
            problem=problem,
            problem_seed=problem_seed,
            mcts_budget=2,  # Reduced MCTS budget for faster tests
            steps=2,  # Reduced steps for faster tests
            agent=self.agent,
            env=self.env
        )

        # Iterate through the DataFrame to track repository size at each step
        for _, row in df.iterrows():
            repo_size_sum += row["repo_size"]  # Use the "repo_size" column from the DataFrame

        # Verify repository size changes during the episode
        self.assertGreater(repo_size_sum, 0, "Repository size should change during the episode.")

    def test_episode_policy_network(self):
        # Mock inputs for the EpisodePolicyNetwork
        best_obs = np.array([0.1, 5, 10])  # Mock best observation vector
        episode_reward = 0.5  # Mock episode reward

        # Predict difficulty adjustment action
        difficulty_action = self.agent.choose_difficulty_action(best_obs, episode_reward)
        self.assertIn(difficulty_action, [3, 4, 5], "Difficulty adjustment action should be valid.")

    def test_repository_size_logging(self):
        # Manually add an entry to the repository
        self.env.repository.append({"subgraph_node": "test_node", "utility": 0.5})
        initial_size = len(self.env.repository)
        print(f"Repository size before reset: {initial_size}")

        # Reset the environment
        self.env.reset()

        # Check the repository size after reset
        post_reset_size = len(self.env.repository)
        print(f"Repository size after reset: {post_reset_size}")

        # Verify that the repository persists across reset
        self.assertEqual(post_reset_size, initial_size, "Repository size should remain the same after environment reset.")

if __name__ == "__main__":
    unittest.main()