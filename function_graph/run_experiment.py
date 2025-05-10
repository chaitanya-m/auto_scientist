import numpy as np
import tensorflow as tf
from keras import layers, models

from envs import FunctionGraphEnv
from agents.mcts import SimpleMCTSAgent

def build_uniform_policy(obs_dim, action_dim):
    """
    Returns a Keras model that maps any observation to uniform probabilities.
    """
    inp = layers.Input(shape=(obs_dim,))
    out = layers.Dense(action_dim, activation="softmax", 
                       kernel_initializer=tf.constant_initializer(1.0/action_dim),
                       bias_initializer=tf.constant_initializer(0.0))(inp)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

def run_simple_experiment(
    phase="basic",
    seed=42,
    mcts_budget=10,
    steps=20,
):
    # 1) Environment setup
    # Instantiate with a placeholder seed (we’ll overwrite below)
    env = FunctionGraphEnv(phase=phase, seed=0)
    max_seeds = env.curriculum.seeds_per_phase
    # Now re-create with the safe seed
    env = FunctionGraphEnv(phase=phase, seed=seed % max_seeds)     # Clamp seed into the precomputed range

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.n

    # 2) Build uniform policy network
    policy_model = build_uniform_policy(obs_dim, action_dim)

    # 3) Instantiate MCTS agent
    agent = SimpleMCTSAgent(env=env, policy_model=policy_model, 
                             search_budget=mcts_budget, c=1.0)

    # 4) Main loop
    metrics = []
    for step in range(steps):
        # a) Run MCTS to pick next action
        action = agent.mcts_search()

        # b) Step environment
        obs, reward, done, truncated, info = env.step(action)

        # c) Collect metrics
        m = {
            "step": step,
            "action": action,
            "mse": env.current_mse,
            "nodes": len(env.composer.nodes),
            "repo_size": len(env.repository),
            "reward": reward
        }
        metrics.append(m)

        # Log
        print(f"[{step:2d}] act={action} | mse={m['mse']:.4f} | nodes={m['nodes']} | repo={m['repo_size']}")

    return metrics

if __name__ == "__main__":
    results = run_simple_experiment(
        phase="basic",
        seed=123,
        mcts_budget=8,
        steps=15
    )

    # Summarize final outcome
    final = results[-1]
    print("\nFinal Candidate:")
    print(f"  MSE:         {final['mse']:.4f}")
    print(f"  # Nodes:     {final['nodes']}")
    print(f"  Repo Size:   {final['repo_size']}")
    print(f"  Last Action: {final['action']}")
