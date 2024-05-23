import random
import concurrent.futures
import numpy as np
from configs import BASE_CONFIG, STREAMS
from environments import *
from agents import *



'''

'''



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

    # The agent is trained on multiple episodes in sequence, each episode corresponding to the stream initialized differently. The Q-table is persistent.
    for _ in range(num_episodes):
        state = env.reset() # The stream is seeded afresh for each episode, thus, each episode corresponds to a different random initialization of the stream
        done = False
        transitions = []  # Store the episode trajectory for Monte Carlo updates


        while not done:  # As long as the episode is not done
            action = agent.select_action(state)
            next_state, reward, done = env.step(action) # A step runs a single stream learning epoch of say 1000 examples
            # Store the transition information for later update (used by both Q-learning and Monte Carlo)

            # Store transition for Monte Carlo updates if necessary
            transitions.append((state, action, reward))

            if isinstance(agent, QLearningAgent):
                # Perform the Q-learning update immediately after the step
                if state is not None and next_state is not None:
                    best_next_action = np.argmax(agent.Q_table[next_state])
                    td_target = reward + agent.gamma * agent.Q_table[next_state][best_next_action] if not done else reward
                    td_error = td_target - agent.Q_table[state][action]
                    agent.Q_table[state][action] += agent.alpha * td_error

            # Move to next state
            state = next_state

        # Get the accuracy and baseline accuracy for this env run
        accuracy = env.cumulative_accuracy / env.current_epoch
        baseline_accuracy = env.cumulative_baseline_accuracy / env.current_epoch

        # Update the agent's Q-table using Monte Carlo updates
        if isinstance(agent, MonteCarloAgent):
            returns = 0
            for (state, action, reward) in reversed(transitions):
                if state is None:
                    continue
                returns = reward + agent.gamma * returns
                agent.visits[state][action] += 1
                alpha = 1 / agent.visits[state][action]
                agent.Q_table[state][action] += alpha * (returns - agent.Q_table[state][action])


def setup_environment_and_train(agent_class, agent_name, num_states, num_actions, num_episodes, config):
    # Since CONFIG and other required variables are not defined in this snippet, 
    # they should be defined elsewhere in the code or passed as arguments to the function.

    # Setup stream factory
    stream_type = config['stream_type']
    stream_factory = StreamFactory(stream_type, config['stream'])

    # Setup model
    ModelClass = model_classes[config['model']]
    model = ModelClass(delta=config['delta_hard'])
    model_baseline = ModelClass(delta=config['delta_hard'])

    # Setup Actions
    actions = config['actions']['delta_move']

    # Setup Environment
    num_samples_per_epoch = config['evaluation_interval']
    num_epochs = config['num_epochs']

    # Train agent
    env = Environment(model, model_baseline, stream_factory, actions, num_samples_per_epoch, num_epochs)
    agent = agent_class(num_states=num_states, num_actions=num_actions)

    train_agent(agent, env, num_episodes)

    return agent


# Top-level function to be used by ProcessPoolExecutor
def run_RL_agents(config):
    try:
        random.seed(config['seed0'])
        np.random.seed(config['seed0'])

        # Define the environment's state and action space sizes and number of episodes
        num_states = NUM_STATES
        num_actions = len(config['actions']['delta_move'])
        num_episodes = config['num_episodes']

        results = []
        results.append(f"config: {config['stream_type']} with params: {config['stream']}")

        # Train Monte Carlo agent
        mc_agent = setup_environment_and_train(MonteCarloAgent, "Monte Carlo", num_states, num_actions, num_episodes, config)
        np.set_printoptions(precision=2)
        results.append(str(mc_agent.Q_table))

        # Train Q-learning agent
        ql_agent = setup_environment_and_train(QLearningAgent, "Q-learning", num_states, num_actions, num_episodes, config)
        np.set_printoptions(precision=2)
        results.append(str(ql_agent.Q_table))

        results.append(f"==========")

        return results

    except Exception as e:
        return [f"An error occurred while processing config {config['stream_type']}: {e}"]


def main():
    # List of configurations
    configs = []
    config = BASE_CONFIG

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
            results = future.result()
            for result in results:
                print(result)


if __name__ == "__main__":
    main()
