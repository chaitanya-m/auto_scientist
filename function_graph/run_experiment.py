import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

from envs import FunctionGraphEnv
from agents.mcts import SimpleMCTSAgent


def build_uniform_policy(obs_dim, action_dim):
    """
    Returns a Keras model that maps any observation to uniform probabilities.
    """
    inp = layers.Input(shape=(obs_dim,))
    out = layers.Dense(
        action_dim, activation="softmax",
        kernel_initializer=tf.constant_initializer(1.0/action_dim),
        bias_initializer=tf.constant_initializer(0.0)
    )(inp)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def run_simple_experiment(
    phase="basic",
    seed=0,
    mcts_budget=10,
    steps=20,
    epsilon_frac=0.01,
):
    """
    Runs a single MCTS experiment on one seed, returning a pandas DataFrame
    of per-step metrics and seed-level summary.
    """
    # Environment setup
    env = FunctionGraphEnv(phase=phase, seed=seed)
    reference_mse = env.reference_mse
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.n

    # Build policy and agent
    policy_model = build_uniform_policy(obs_dim, action_dim)
    agent = SimpleMCTSAgent(
        env=env,
        policy_model=policy_model,
        search_budget=mcts_budget,
        c=1.0
    )

    # Run loop, collect metrics
    metrics = []
    reuse_count = 0
    threshold = reference_mse * (1.0 + epsilon_frac)
    reached_threshold = False
    step_to_epsilon = None

    for step in range(steps):
        prev_best = env.best_mse

        # Choose action via MCTS
        action = agent.mcts_search()

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Compute additional metrics
        did_reuse = (action == 2)
        if did_reuse:
            reuse_count += 1

        improvement = max(0.0, prev_best - env.current_mse)

        if not reached_threshold and env.current_mse <= threshold:
            reached_threshold = True
            step_to_epsilon = step

        m = {
            "seed": seed,
            "step": step,
            "action": action,
            "did_reuse": did_reuse,
            "cumulative_reuse": reuse_count,
            "mse": env.current_mse,
            "improvement": improvement,
            "nodes": len(env.composer.nodes),
            "repo_size": len(env.repository),
            "improvement_count": env.improvement_count,
            "deletion_count": env.deletion_count,
            "reward": reward,
            "reference_mse": reference_mse,
            "epsilon_threshold": threshold,
        }
        metrics.append(m)

        print(
            f"[Seed {seed} | Step {step:2d}] act={action} "
            f"| reuse={reuse_count} | mse={m['mse']:.4f} "
            f"| nodes={m['nodes']} | repo={m['repo_size']}"
        )

    df = pd.DataFrame(metrics)
    # Seed-level summary
    final = df.iloc[-1]
    summary = {
        "seed": seed,
        "steps_to_epsilon": step_to_epsilon if reached_threshold else np.nan,
        "total_reuse": reuse_count,
        "final_mse": final["mse"],
        "final_nodes": final["nodes"],
        "final_repo_size": final["repo_size"],
    }
    return df, summary


def main(
    phase="basic",
    mcts_budget=8,
    steps=15,
    epsilon_frac=0.01,
    output_dir="results",
):
    """
    Runs experiments across all seeds in the given phase,
    printing terminal summaries and saving a single CSV of all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine seeds count
    env0 = FunctionGraphEnv(phase=phase, seed=0)
    max_seeds = env0.curriculum.seeds_per_phase

    all_dfs = []
    summaries = []

    for seed in range(max_seeds):
        df, summary = run_simple_experiment(
            phase=phase,
            seed=seed,
            mcts_budget=mcts_budget,
            steps=steps,
            epsilon_frac=epsilon_frac,
        )
        summaries.append(summary)
        all_dfs.append(df)

    # Combine and save metrics
    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(output_dir, f"results_{phase}.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined metrics to {csv_path}\n")

    # Convert summaries to DataFrame
    sum_df = pd.DataFrame(summaries)

    # Terminal summary
    print("=== Experiment Summary ===")
    print(f"Phase: {phase}")
    print(f"Seeds: {max_seeds}")
    print(f"Steps per run: {steps}")
    print("\n-- Steps to Epsilon (ε threshold) --")
    print(sum_df['steps_to_epsilon'].describe())

    print("\n-- Total Reuse Actions --")
    print(sum_df['total_reuse'].describe())

    print("\n-- Final MSE --")
    print(sum_df['final_mse'].describe())

    print("\n-- Final Node Counts --")
    print(sum_df['final_nodes'].describe())

    print("\n-- Final Repository Sizes --")
    print(sum_df['final_repo_size'].describe())

if __name__ == "__main__":
    main(
        phase="basic",
        mcts_budget=8,
        steps=15,
        epsilon_frac=0.01,
        output_dir="results"
    )
