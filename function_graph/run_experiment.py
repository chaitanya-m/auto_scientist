# run_experiment.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

from data_gen.problems import AutoencoderProblem, Problem
from envs import FunctionGraphEnv

def build_uniform_policy(obs_dim: int, action_dim: int) -> models.Model:
    """
    Returns a Keras model that maps any observation to uniform probabilities.
    """
    inp = layers.Input(shape=(obs_dim,))
    out = layers.Dense(
        action_dim,
        activation="softmax",
        kernel_initializer=tf.constant_initializer(1.0/action_dim),
        bias_initializer=tf.constant_initializer(0.0)
    )(inp)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

def run_simple_experiment(
    problem: Problem,
    seed: int = 0,
    mcts_budget: int = 10,
    steps: int = 20,
) -> (pd.DataFrame, dict):
    """
    Runs a single MCTS experiment on one Problem instance,
    returning a DataFrame of per-step metrics and a summary dict.
    """
    # 1) Environment and agent setup
    env = FunctionGraphEnv(problem=problem, seed=seed)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.n

    policy_model = build_uniform_policy(obs_dim, action_dim)
    from agents.mcts import SimpleMCTSAgent
    agent = SimpleMCTSAgent(env=env, policy_model=policy_model,
                             search_budget=mcts_budget, c=1.0)

    # 2) Run the search loop
    metrics = []
    reuse_count = 0

    for step in range(steps):
        prev_best = env.best_mse
        action = agent.mcts_search()
        obs, reward, done, truncated, info = env.step(action)

        did_reuse = (action == 2)
        if did_reuse:
            reuse_count += 1

        improvement = max(0.0, prev_best - env.current_mse)

        metrics.append({
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
            "reward": reward
        })

        print(
            f"[Seed {seed} | Step {step:2d}] "
            f"act={action} | reuse={reuse_count} "
            f"| mse={env.current_mse:.4f} "
            f"| nodes={len(env.composer.nodes)} "
            f"| repo={len(env.repository)}"
        )

    df = pd.DataFrame(metrics)

    # 3) Compute summary for this run
    eps = 0.01 * problem.reference_mse
    steps_to_eps = df[df.mse <= problem.reference_mse + eps].step.min()
    summary = {
        "seed": seed,
        "steps_to_epsilon": float(steps_to_eps) if not np.isnan(steps_to_eps) else np.nan,
        "total_reuse": reuse_count,
        "final_mse": float(df.mse.iloc[-1]),
        "final_nodes": int(df.nodes.iloc[-1]),
        "final_repo_size": int(df.repo_size.iloc[-1])
    }

    return df, summary

def main(
    phase: str = "basic",
    mcts_budget: int = 8,
    steps: int = 15,
    output_dir: str = "results",
):
    """
    Runs experiments across all seeds in the given phase,
    saving a single CSV and printing terminal summaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a prototype autoencoder to determine how many seeds exist.
    base_autoencoder = AutoencoderProblem(phase=phase, seed=0)
    num_seeds = base_autoencoder.seeds_per_phase

    all_dfs = []
    summaries = []

    for seed in range(num_seeds):
        # instantiate a fresh problem for each seed
        problem = AutoencoderProblem(phase=phase, seed=seed)
        df, summary = run_simple_experiment(
            problem=problem,
            seed=seed,
            mcts_budget=mcts_budget,
            steps=steps
        )
        all_dfs.append(df)
        summaries.append(summary)

    # Concatenate all runs into one CSV.
    master_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(output_dir, f"results_{phase}.csv")
    master_df.to_csv(csv_path, index=False)
    print(f"Saved detailed metrics to {csv_path}")

    # Print terminal summary
    sum_df = pd.DataFrame(summaries)
    print("\n=== Experiment Summary ===")
    print(f"Phase: {phase}")
    print(f"Seeds: {num_seeds}")
    print(f"Steps per run: {steps}\n")

    print("-- Steps to Epsilon (ε threshold) --")
    print(sum_df.steps_to_epsilon.describe(), "\n")
    print("-- Total Reuse Actions --")
    print(sum_df.total_reuse.describe(), "\n")
    print("-- Final MSE --")
    print(sum_df.final_mse.describe(), "\n")
    print("-- Final Node Counts --")
    print(sum_df.final_nodes.describe(), "\n")
    print("-- Final Repository Sizes --")
    print(sum_df.final_repo_size.describe(), "\n")

if __name__ == "__main__":
    main()
