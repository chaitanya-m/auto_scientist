import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from typing import Type
from data_gen.problems import Problem
from data_gen.curriculum import Curriculum

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
    problem_seed: int = 0,
    mcts_budget: int = 10,
    steps: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """
    Runs a single MCTS experiment on one Problem instance,
    returning a DataFrame of per-step metrics and a summary dict.
    """
    from envs import FunctionGraphEnv
    from agents.mcts import SimpleMCTSAgent

    # 1) Environment and agent setup
    env = FunctionGraphEnv(problem=problem, seed=problem_seed)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.n

    policy_model = build_uniform_policy(obs_dim, action_dim)
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
            "seed": problem_seed,
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
            f"[Seed {problem_seed} | Step {step:2d}] "
            f"act={action} | reuse={reuse_count} "
            f"| mse={env.current_mse:.4f} "
            f"| nodes={len(env.composer.nodes)} "
            f"| repo={len(env.repository)}"
        )

    df = pd.DataFrame(metrics)

    # 3) Compute summary for this run using aggregated per-step metrics.
    # Define epsilon threshold as reference mse + 1% of reference mse.
    eps_threshold = problem.reference_mse + (0.01 * problem.reference_mse)
    try:
        steps_to_eps = df.loc[df["mse"] <= eps_threshold, "step"].iloc[0]
    except IndexError:
        steps_to_eps = np.nan

    summary = {
        "seed": problem_seed,
        "steps_to_epsilon": float(steps_to_eps),
        "total_reuse": int(df["cumulative_reuse"].iloc[-1]),
        "final_mse": float(df["mse"].iloc[-1]),
        "final_nodes": int(df["nodes"].iloc[-1]),
        "final_repo_size": int(df["repo_size"].iloc[-1])
    }

    return df, summary

def default_curriculum(problem_cls: Type[Problem], phase: int, num_problems: int = 2) -> Curriculum:
    """
    Returns a default Curriculum instance using the provided Problem class.
    The default generator calls seeded_problem_variations on the problem_cls.
    
    Arguments:
      problem_cls  : a Problem class implementing seeded_problem_variations.
      phase        : the phase number to be used for problem generation.
      num_problems : number of problems (seeds) to generate.
    """
    def default_generator():
        return problem_cls.seeded_problem_variations(phase, num_problems)
    return Curriculum(default_generator)

def run_experiments(curriculum : Curriculum, mcts_budget: int, steps: int, output_dir: str):
    """
    Runs experiments over all problems in the given Curriculum.

    Arguments:
      curriculum : a Curriculum object that yields Problem instances.
      mcts_budget: search budget for each experiment.
      steps      : number of steps per experiment.
      output_dir : path to save CSV results.
      
    If needed, a default curriculum can be generated using default_curriculum().
    """
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []
    summaries = []
    
    for problem in curriculum:
        df, summary = run_simple_experiment(
            problem=problem,
            problem_seed=problem.problem_seed,
            mcts_budget=mcts_budget,
            steps=steps
        )
        # Inject the phase information from the curriculum.
        summary["phase"] = getattr(curriculum, "phase", "N/A")
        all_dfs.append(df)
        summaries.append(summary)
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    # Optionally, you may want to include curriculum-specific details in the output filename.
    csv_path = os.path.join(output_dir, f"results_curriculum.csv")
    master_df.to_csv(csv_path, index=False)
    print(f"Saved detailed metrics to {csv_path}")
    
    print_experiment_summary(summaries)

def print_experiment_summary(summaries):
    sum_df = pd.DataFrame(summaries)
    print("\n=== Experiment Summary ===")
    print(f"Total Problems: {len(summaries)}")
    print("-- Steps to Epsilon (ε threshold) --")
    print(sum_df.steps_to_epsilon.describe(), "\n")
    print("-- Total Reuse Actions --")
    print(sum_df.total_reuse.describe(), "\n")
    print("-- Final MSE --")
    print(sum_df.final_mse.describe(), "\n")