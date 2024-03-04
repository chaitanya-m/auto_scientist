import random


from configs import CONFIG
from environments import *
from agents import *


def train_agent(agent, env, num_episodes):
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episodes = []  # Store the episode for Monte Carlo updates

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Store the transition information for later update (used by both Q-learning and Monte Carlo)
            episodes.append((state, action, reward))

            # Move to the next state
            state = next_state
        
        # Now perform the update outside the Agent classes
        if isinstance(agent, QLearningAgent):
            for episode_index, (state, action, reward) in enumerate(episodes[:-1]):  # Skip the last state since it has no next_state
                if state is None:
                    continue
                next_state = episodes[episode_index + 1][0]  # The state of the next transition
                best_next_action = np.argmax(agent.Q_table[next_state])
                td_target = reward + agent.gamma * agent.Q_table[next_state][best_next_action]
                td_error = td_target - agent.Q_table[state][action]
                agent.Q_table[state][action] += agent.alpha * td_error

        elif isinstance(agent, MonteCarloAgent):
            returns = 0
            for (state, action, reward) in reversed(episodes):
                if state is None:
                    continue
                returns = reward + agent.gamma * returns
                agent.visits[state][action] += 1
                alpha = 1 / agent.visits[state][action]
                agent.Q_table[state][action] += alpha * (returns - agent.Q_table[state][action])


def setup_environment_and_train(agent_class, agent_name, num_states, num_actions, num_episodes):
    # Since CONFIG and other required variables are not defined in this snippet, 
    # they should be defined elsewhere in the code or passed as arguments to the function.

    # Setup stream factory
    stream_type = CONFIG['stream_type']
    stream_factory = StreamFactory(stream_type, CONFIG['streams'][stream_type])

    # Setup model
    ModelClass = model_classes[CONFIG['model']]
    model = ModelClass(delta=CONFIG['delta_hard'])
    model_baseline = ModelClass(delta=CONFIG['delta_hard'])

    # Setup Actions
    actions = CONFIG['actions']['delta_move']

    # Setup Environment
    num_samples_per_epoch = CONFIG['evaluation_interval']
    num_epochs = CONFIG['num_epochs']


    # Train agent
    env = Environment(model, model_baseline, stream_factory, actions, num_samples_per_epoch, num_epochs)
    agent = agent_class(num_states=num_states, num_actions=num_actions)

    train_agent(agent, env, num_episodes)

    print(f"Q-table ({agent_name}):")
    # print only 4 significant digits
    np.set_printoptions(precision=4)
    print(agent.Q_table)

def main():
    random.seed(CONFIG['seed0'])
    np.random.seed(CONFIG['seed0'])

    # Define the environment's state and action space sizes and number of episodes
    num_states = 25
    num_actions = len(CONFIG['actions']['delta_move']) 
    num_episodes = CONFIG['num_episodes']

    # Train Monte Carlo agent
    setup_environment_and_train(MonteCarloAgent, "Monte Carlo", num_states, num_actions, num_episodes)

    # Train Q-learning agent
    setup_environment_and_train(QLearningAgent, "Q-learning", num_states, num_actions, num_episodes)


if __name__ == "__main__":
    main()


