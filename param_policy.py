import random

from configs import BASE_CONFIG, STREAMS
from environments import *
from agents import *

def train_agent(agent, env, num_episodes):

    # 
    for _ in range(num_episodes):
        state = env.reset() # The stream is seeded afresh for each episode, thus, each episode corresponds to a different random initialization of the stream
        done = False
        transitions = []  # Store the episode trajectory for Monte Carlo updates

        while not done: # as long as the episode is not done, i.e. as long as the stream has examples
            action = agent.select_action(state)
            next_state, reward, done = env.step(action) # A step runs a single stream learning epoch of say 1000 examples
            # Store the transition information for later update (used by both Q-learning and Monte Carlo)
            transitions.append((state, action, reward))

            # Move to the next state
            state = next_state
        
        # Now perform the update outside the Agent classes
        # Look back at previous transitions --- comprising states, actions, and rewards --- to update 
        # understanding of how valuable different actions are when taken in different states
        if isinstance(agent, QLearningAgent):
            # Loop over each transition in the list of transitions, excluding the last one
            for transition_index, (state, action, reward) in enumerate(transitions[:-1]):  # Skip the last state since it has no next_state                
                if state is None: # If the state is None, skip the current iteration
                    continue

                next_state = transitions[transition_index + 1][0] 
                # Get the state of the next transition, as per the trajectory taken  

                best_next_action = np.argmax(agent.Q_table[next_state]) 
                # From the Q_table, lookup the action that has the maximum Q-value for that next state. That is, the best action to have taken in 
                # the next_state in retrospect is selected as the one with the highest Q-value in the Q-table for that state.
                # Note that when multiple actions have the same Q-value, the first one is selected, so there may be a bias in the selection towards
                # earlier actions as the table starts all zeroed.

                td_target = reward + agent.gamma * agent.Q_table[next_state][best_next_action] 
                # Calculate the target Q-value using the reward and the discounted Q-value of the best next action
                # The temporal difference (TD) target is calculated using the (retrospective) reward received for the 
                # "current" action plus the discounted value of the best possible action in the next state. This forms the basis of the 
                # Q-learning update rule and reflects the expected long-term return.

                td_error = td_target - agent.Q_table[state][action] 
                # Calculate the difference between the target and the current Q-value
                # The TD error (or difference) is the difference between the calculated TD target and the 
                # "currently" estimated Q-value for the state-action pair.

                agent.Q_table[state][action] += agent.alpha * td_error 
                # Update the Q-value for the current state and action
                # The Q-value for the current state and action is updated by moving it towards the TD target. 
                # The learning rate alpha determines how much the new information overrides the old information.

        elif isinstance(agent, MonteCarloAgent):
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

    print(f"Q-table ({agent_name}), ({stream_type}):")
    # print only 4 significant digits
    np.set_printoptions(precision=2)
    print(agent.Q_table)


def main():
    # List of configurations
    configs= []
    config = BASE_CONFIG

    # Loop through the STREAMS dictionaries in the configs and replace the BASE_CONFIG with the STREAMS one at a time
   
    for stream_config in STREAMS:
            new_config = config.copy()
            new_config['stream_type'] = stream_config['stream_type']
            new_config['stream'] = stream_config['stream']
            configs.append(new_config)

    for config in configs:
        random.seed(config['seed0'])
        np.random.seed(config['seed0'])

        # Define the environment's state and action space sizes and number of episodes
        num_states = NUM_STATES
        num_actions = len(config['actions']['delta_move']) 
        num_episodes = config['num_episodes']

        # Train Monte Carlo agent
        setup_environment_and_train(MonteCarloAgent, "Monte Carlo", num_states, num_actions, num_episodes, config)

        # Train Q-learning agent
        setup_environment_and_train(QLearningAgent, "Q-learning", num_states, num_actions, num_episodes, config)

if __name__ == "__main__":
    main()


