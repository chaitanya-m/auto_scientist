import random
import concurrent.futures
import numpy as np
import pandas as pd
from configs import BASE_CONFIG, STREAMS
from environments import *
from agents import *
from actions import ModifyAlgorithmStateAction
from itertools import combinations

def generate_combinations(size):
    indices = list(range(size))
    
    # Generate combinations of all lengths from 1 to size
    for r in range(1, size + 1):
        for comb in combinations(indices, r):
            yield list(comb)

def train_agent(agent, env, num_episodes):
    '''
    Train the agent using the given environment for the specified number of episodes.
    Either Q-learning or Monte Carlo updates are used based on the agent type.

    Q-learning:

    From the Q_table, lookup the action that has the maximum Q-value for that next state. That is, the best action to have taken in 
    the next_state in retrospect is selected as the one with the highest Q-value in the Q-table for that state.
    Note that when multiple actions have the same Q-value, the first one is selected, so there may be a bias in the selection towards
    earlier actions as the table starts all zeroed.

    # Calculate the target Q-value using the reward and the discounted Q-value of the best next action
    # The temporal difference (TD) target is calculated using the (retrospective) reward received for the 
    # "current" action plus the discounted value of the best possible action in the next state. This forms the basis of the 
    # Q-learning update rule and reflects the expected long-term return.

    # Calculate the difference between the target and the current Q-value
    # The TD error (or difference) is the difference between the calculated TD target and the 
    # "currently" estimated Q-value for the state-action pair.

    # Update the Q-value for the current state and action
    # The Q-value for the current state and action is updated by moving it towards the TD target. 
    # The learning rate alpha determines how much the new information overrides the old information.

    '''

    episode_accuracies = []
    episode_baseline_accuracies = []

    # The agent is trained on multiple episodes in sequence, each episode corresponding to the stream initialized differently. The Q-table is persistent.
    for _ in range(num_episodes):
        state_index = env.reset() # The stream is seeded afresh for each episode, thus, each episode corresponds to a different random initialization of the stream
        # An algorithm is also randomly picked from the design space
        done = False
        transitions = []  # Store the episode trajectory for Monte Carlo updates

        while not done:  # As long as the episode is not done
            action_index = agent.select_action(state_index)
            new_state_index, reward, done = env.step(action_index) # A step runs a single stream learning epoch of say 1000 examples
            # Store the transition information for later update (used by both Q-learning and Monte Carlo)

            # Store transition for Monte Carlo updates if necessary. Note that step should've already updated the state.
            transitions.append((state_index, action_index, reward))

            if isinstance(agent, QLearningAgent):
                # Perform the Q-learning update immediately after the step
                # On policy because we're using the best possible next action
                if state_index is not None and new_state_index is not None:
                    best_next_action = np.argmax(agent.Q_table[new_state_index])
                    td_target = reward + agent.gamma * agent.Q_table[new_state_index][best_next_action] if not done else reward
                    td_error = td_target - agent.Q_table[state_index][action_index]
                    agent.Q_table[state_index][action_index] += agent.alpha * td_error

            # Update the state index for the next iteration
            state_index = new_state_index

        # Update the agent's Q-table using Monte Carlo updates at the end of an episode
        # Off policy because we're using the transition as is, not the best possible next action
        if isinstance(agent, MonteCarloAgent):
            returns = 0
            for (state_index, action_index, reward) in reversed(transitions):
                if state_index is None:
                    continue
                returns = reward + agent.gamma * returns
                agent.visits[state_index][action_index] += 1
                #alpha = 1 / agent.visits[state_index][action_index]
                alpha = agent.alpha_mc_decay ** agent.visits[state_index][action_index]  # Decay alpha
                agent.Q_table[state_index][action_index] += alpha * (returns - agent.Q_table[state_index][action_index])

        # Get the accuracy and baseline accuracy for this episode
        accuracy = env.cumulative_accuracy / env.current_epoch
        baseline_accuracy = env.cumulative_baseline_accuracy / env.current_epoch
        episode_accuracies.append(accuracy)
        episode_baseline_accuracies.append(baseline_accuracy)
    return episode_accuracies, episode_baseline_accuracies


def setup_environment_and_train(agent_class, agent_name, num_states, num_episodes, config):
    # Since CONFIG and other required variables are not defined in this snippet, 
    # they should be defined elsewhere in the code or passed as arguments to the function.

    # Every time the environment is setup, we have to create a new state
    # We may want to start in various random states across runs to determine convergence
    state = AlgorithmState([0]*config['algo_vec_len'])

    # Setup Actions
    # actions are now all possible combinations of flips of the binary values in algorithm state vector
    # Plus the no-op action
    # This means that for m indices, there are 2^m+1 possible actions
    # We need to add all the possible actions to the environment by creating ModifyAlgorithmStateAction 
    # with all possible indices for config['algo_vec_len'] indices

    actions = [ModifyAlgorithmStateAction(indices) for indices in generate_combinations(config['algo_vec_len'])]
    actions.append(ModifyAlgorithmStateAction([]))  # Add the no-op action

    binary_design_space = {
    "_reevaluate_best_split": ["reevaluate_best_split_removed", "original_reevaluate_best_split"],
    "_attempt_to_split": ["attempt_to_split_removed", "original_attempt_to_split"]
    }

    # Setup stream factory
    stream_type = config['stream_type']
    stream_factory = StreamFactory(stream_type, config['stream'])

    # Setup model
    ModelClass = model_classes[config['model_class']]
    model = ModelClass(delta=config['delta_hard'])
    BaselineModelClass = model_classes[config['baseline_model']]
    model_baseline = BaselineModelClass(delta=config['delta_hard'])


    # Setup Environment
    num_samples_per_epoch = config['evaluation_interval']
    num_epochs = config['num_epochs']

    if config['debug']:
        print("\nDebug: Actions are all null\n")
        test_null_actions = [ModifyAlgorithmStateAction([]) for _ in generate_combinations(config['algo_vec_len'])]
        test_null_actions.append(ModifyAlgorithmStateAction([]))
        actions = test_null_actions

    # Train agent
    env = Environment(state, actions, binary_design_space ,model, model_baseline, stream_factory, num_samples_per_epoch, num_epochs)
    agent = agent_class(num_states=num_states, num_actions=len(actions), config=config)

    accuracies, baseline_accuracies = train_agent(agent, env, num_episodes)

    # Now create a dataframe with the results if it already doesn't exist

    return accuracies, baseline_accuracies, agent.Q_table


# Top-level function to be used by ProcessPoolExecutor
def run_RL_agents(config):
    try:
        random.seed(config['seed0'])
        np.random.seed(config['seed0'])

        # Define the environment's context and number of episodes
        num_states = 2 ** config['algo_vec_len']
        num_episodes = config['num_episodes']

        result_qtables = []
        result_qtables.append(f"config: {config['stream_type']} with params: {config['stream']}")

        # The dataframe has columns for the episode number, config['stream_type'], config['stream'], accuracy, and baseline accuracy
        mc_result_accuracies_df = pd.DataFrame(columns=['episode', 'agent_type','stream_type', 'stream', 'accuracy', 'baseline_accuracy'])
        ql_result_accuracies_df = pd.DataFrame(columns=['episode', 'agent_type','stream_type', 'stream', 'accuracy', 'baseline_accuracy'])

        # Train Monte Carlo agent
        mc_accuracies, mc_baseline_accuracies, mc_qtable = setup_environment_and_train(MonteCarloAgent, "Monte Carlo", num_states, num_episodes, config)

        # Train Q-learning agent
        ql_accuracies, ql_baseline_accuracies, ql_qtable = setup_environment_and_train(QLearningAgent, "Q-learning", num_states, num_episodes, config)

        # Add accuracies to the dataframe. 
        mc_temp_data = []
        ql_temp_data = []

        for episode, (accuracy, baseline_accuracy) in enumerate(zip(ql_accuracies, ql_baseline_accuracies)):
            # Create a dictionary for each iteration and add to list
            ql_temp_data.append({
                'episode': episode,
                'agent_type': 'Q-learning',
                'stream_type': config['stream_type'],
                'stream': config['stream'],
                'accuracy': accuracy,
                'baseline_accuracy': baseline_accuracy
            })

        for episode, (accuracy, baseline_accuracy) in enumerate(zip(mc_accuracies, mc_baseline_accuracies)):
            # Create a dictionary for each iteration and add to list
            mc_temp_data.append({
                'episode': episode,
                'agent_type': 'Monte Carlo',
                'stream_type': config['stream_type'],
                'stream': config['stream'],
                'accuracy': accuracy,
                'baseline_accuracy': baseline_accuracy
            })

        # Convert list of dictionaries to DataFrame
        #new_entries_df 
        mc_result_accuracies_df = pd.DataFrame(mc_temp_data)
        ql_result_accuracies_df = pd.DataFrame(ql_temp_data)

        # Concatenate the new entries to the existing DataFrame
        #result_accuracies_df = pd.concat([result_accuracies_df, new_entries_df], ignore_index=True)

        # Add Q-tables to the results
        np.set_printoptions(precision=2)
        result_qtables.append(str(mc_qtable))
        np.set_printoptions(precision=2)
        result_qtables.append(str(ql_qtable))
        result_qtables.append(f"==========")

        #print(mc_result_accuracies_df)

        return result_qtables, mc_result_accuracies_df, ql_result_accuracies_df

    except Exception as e:
        return [f"An error occurred while processing config {config['stream_type']}: {e}"]


def main():
    # List of configurations
    configs = []
    config = BASE_CONFIG

    # Initialize dataframes for concatenating results
    all_result_mc_df = pd.DataFrame()
    all_result_ql_df = pd.DataFrame()

    all_result_qtables = []

    # Loop through the STREAMS dictionaries in the configs and replace the BASE_CONFIG with the STREAMS one at a time
    for stream_config in STREAMS:
        new_config = config.copy()
        new_config['stream_type'] = stream_config['stream_type']
        new_config['stream'] = stream_config['stream']
        configs.append(new_config)

    # Using ProcessPoolExecutor to run tasks in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all configurations as separate tasks
        futures = [executor.submit(run_RL_agents, config) for config in configs]
        
        # Collect results in the order they were submitted
        for future in concurrent.futures.as_completed(futures):
            result_q_tables, result_mc_accuracies_df, result_ql_accuracies_df = future.result()

            # Concatenate the results into the respective dataframes
            all_result_mc_df = pd.concat([all_result_mc_df, result_mc_accuracies_df])
            all_result_ql_df = pd.concat([all_result_ql_df, result_ql_accuracies_df])
            all_result_qtables.extend(result_q_tables)

    # Write the concatenated dataframes to CSV files
    all_result_mc_df.to_csv('all_result_mc_df.csv', index=False)
    all_result_ql_df.to_csv('all_result_ql_df.csv', index=False)

    # Write the concatenated list of result_q_tables to a text file
    with open('all_result_q_tables.txt', 'w') as f:
        for q_table in all_result_qtables:
            f.write(q_table + '\n')


if __name__ == "__main__":
    main()
