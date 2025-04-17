# exp.py

import exp_setup  # sets up sys.path so packages can be found

from agents.mcts import SimpleMCTSAgent

def run_experiment():
    # Instantiate the MCTS agent.
    agent = SimpleMCTSAgent()

    # Define the search budget (number of iterations) for our experiment.
    search_budget = 20  # You can adjust this for a longer search if desired.
    exploration_constant = 1.41  # Use the default or adjust if needed.
    
    print("=== Starting MCTS Experiment ===")
    print(f"Search budget: {search_budget} iterations")
    
    # Run the MCTS search.
    final_state = agent.mcts_search(search_budget=search_budget, exploration_constant=exploration_constant)
    
    # Retrieve final candidate performance (MSE) from the final state.
    candidate_mse = final_state.get("performance", None)
    
    # Print out key experimental metrics.
    print("\n=== Experiment Results ===")
    print(f"Final Candidate Encoder MSE: {candidate_mse}")
    print(f"Final Repository Size: {len(agent.repository)}")
    print(f"Total Deletion Actions: {agent.deletion_count}")
    print(f"Total Improvement Events: {agent.improvement_count}")
    
    # Optionally, print final state action history to see if 'add_from_repository' was used.
    action_history = final_state.get("graph_actions", [])
    print(f"Final Action History: {action_history}")

if __name__ == '__main__':
    run_experiment()
