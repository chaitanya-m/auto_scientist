import numpy as np
from data_gen.categorical_classification import DataSchemaFactory

class RLEnvironment:
    def __init__(self, total_steps=1000, num_instances_per_step=100, seed=0):
        """
        total_steps: Total number of steps in the episode.
        num_instances_per_step: Number of data points provided at each step.
        seed: Seed for generating the fixed data distribution.
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.num_instances_per_step = num_instances_per_step
        self.seed = seed

        # Create a fixed classification schema (binary classification with 2 features, 2 categories).
        self.factory = DataSchemaFactory()
        self.schema = self.factory.create_schema(
            num_features=2,
            num_categories=2,
            num_classes=2,
            random_seed=self.seed
        )
        # Generate the dataset once; all steps use data drawn from this distribution.
        self.dataset = self.schema.generate_dataset(
            num_instances=self.num_instances_per_step,
            random_seed=123  # fixed seed for dataset generation within the episode
        )

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # The state includes the current step and the dataset.
        return {
            "step": self.current_step,
            "dataset": self.dataset  # DataFrame from categorical_classification.py
        }

    def valid_actions(self):
        # For now, we keep the same two actions: "reuse" or "new".
        return ["reuse", "new"]

    def step(self, action):
        """
        At each step, the environment provides the same dataset.
        The action (e.g., "reuse" or "new") will eventually affect performance and cost.
        Here we simply return the current state, a dummy reward, and whether the episode is done.
        """
        reward = 0  # Placeholder for reward calculation.
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.total_steps
        return next_state, reward, done

# Dummy agent that randomly selects among valid actions.
class DummyAgent:
    def __init__(self):
        self.actions_history = []

    def choose_action(self, state, valid_actions):
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

if __name__ == "__main__":
    env = RLEnvironment(total_steps=1000, num_instances_per_step=100, seed=0)
    agent = DummyAgent()
    actions, rewards = run_episode(env, agent)
    print(f"Total steps: {len(actions)}")
    print(f"Actions taken (first 10): {actions[:10]} ...")
    state = env._get_state()
    print("Dataset sample:")
    print(state["dataset"].head())
