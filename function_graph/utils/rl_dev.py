#rl_dev.py

import numpy as np
import pandas as pd
from data_gen.categorical_classification import DataSchemaFactory

# -----------------------
# Experiment Parameters
# -----------------------
NUM_EPISODES = 10          # 10 episodes, seeds 0 to 9
ROUNDS_PER_EPISODE = 10    # 10 rounds per episode
NUM_INSTANCES = 100        # Instances per round
NUM_FEATURES = 2
NUM_CATEGORIES = 2
NUM_CLASSES = 2

# -----------------------
# Agent Strategies
# -----------------------
def agent_A_strategy(df):
    # Predict all zeros
    return np.zeros(len(df), dtype=int)

def agent_B_strategy(df):
    # Predict randomly from {0, 1}
    rng = np.random.default_rng()
    return rng.choice([0, 1], size=len(df))

# -----------------------
# Reward Function
# -----------------------
def compute_reward(acc):
    # Penalize if accuracy < 50%; else reward equals accuracy
    return -1 if acc < 0.5 else acc

# -----------------------
# Experiment Runner
# -----------------------
def run_experiment():
    experiment_log = []  # To store results for analysis

    factory = DataSchemaFactory()

    # Loop over episodes, each with a unique seed for data generation.
    for episode_seed in range(NUM_EPISODES):
        # Create a binary classification schema and dataset generator.
        schema = factory.create_schema(num_features=NUM_FEATURES,
                                       num_categories=NUM_CATEGORIES,
                                       num_classes=NUM_CLASSES,
                                       random_seed=episode_seed)
        # All rounds in this episode use the same data distribution.
        df_data = schema.generate_dataset(num_instances=NUM_INSTANCES, random_seed=123)

        episode_results = {"episode_seed": episode_seed, "rounds": []}

        for round_num in range(ROUNDS_PER_EPISODE):
            # Agents apply their strategies
            preds_A = agent_A_strategy(df_data)
            preds_B = agent_B_strategy(df_data)

            # Ground truth labels are in the "label" column
            true_labels = df_data["label"].to_numpy(dtype=int)

            # Compute accuracies
            acc_A = np.mean(preds_A == true_labels)
            acc_B = np.mean(preds_B == true_labels)

            # Compute rewards
            reward_A = compute_reward(acc_A)
            reward_B = compute_reward(acc_B)

            round_result = {
                "round": round_num,
                "acc_A": acc_A,
                "acc_B": acc_B,
                "reward_A": reward_A,
                "reward_B": reward_B
            }
            episode_results["rounds"].append(round_result)

            # (Agents may update strategies here based on reward feedback)
            # For this simple experiment, we use fixed strategies.

        experiment_log.append(episode_results)

    return experiment_log

# -----------------------
# Run and Analyze Experiment
# -----------------------
if __name__ == "__main__":
    log = run_experiment()
    
    # Simple analysis: Print average rewards per episode.
    for ep in log:
        rewards_A = [r["reward_A"] for r in ep["rounds"]]
        rewards_B = [r["reward_B"] for r in ep["rounds"]]
        avg_reward_A = np.mean(rewards_A)
        avg_reward_B = np.mean(rewards_B)
        print(f"Episode Seed {ep['episode_seed']}: Agent A avg reward = {avg_reward_A:.3f}, Agent B avg reward = {avg_reward_B:.3f}")
    
    # Optionally, further analysis of subgraph usage and strategy changes can be added.
