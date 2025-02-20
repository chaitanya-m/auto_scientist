# utils/rl.py
from graph.composer import GraphTransformer

def run_episode(env, agents, seed=0, steps=0, schema=None, step_callback=None):
    """
    Runs an episode of the environment for a given number of steps.
    """
    # Reset environment to a fresh state.
    state = env.reset(seed=seed, new_schema=schema)
    
    # Compile models for each agent if needed.
    # Now we get agent graphmodels from the state.
    for agent_id, agent_state in state.agents_states.items():
        composer, model = agent_state.graphmodel
        if not hasattr(model, "optimizer"):
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Prepare history dictionaries.
    actions_history = {agent_id: [] for agent_id in state.agents_states}
    rewards_history = {agent_id: [] for agent_id in state.agents_states}
    accuracies_history = {agent_id: [] for agent_id in state.agents_states}
    debug_counter = {agent_id: 0 for agent_id in state.agents_states}
    
    # Main loop over steps.
    for step_count in range(steps):
        # Generate new data and update the environment state.
        state, rewards = env.step()
        for agent_id in agents:         # Add rewards to history.
            rewards_history[agent_id].append(rewards[agent_id])

        # Optional: let the user update valid actions or policies.
        if step_callback is not None:
            step_callback(step_count, state, agents, env)
        
        # Each agent picks an action.
        chosen_actions = {}
        for agent_id in agents:
            agent_state = state.agents_states[agent_id]
            action = agents[agent_id].choose_action(agent_state, step_count)
            chosen_actions[agent_id] = action
            actions_history[agent_id].append(action)
        
        # Apply any "add_abstraction" actions.
        for agent_id, action in chosen_actions.items():
            if action == "add_abstraction":
                debug_counter[agent_id] += 1
                print(f"DEBUG: {agent_id} adds abstraction {debug_counter[agent_id]} times")
                learned_abstraction = state.repository["learned_abstraction"]
                # Get the agent's current graphmodel from the state.
                composer, model = state.agents_states[agent_id].graphmodel
                transformer = GraphTransformer(composer)
                new_model = transformer.add_abstraction_node(
                    abstraction_node=learned_abstraction,
                    chosen_subset=["input"],
                    outputs=["output"],
                    remove_prob=1.0
                )
                # Update the agent's graphmodel via the state.
                state.agents_states[agent_id].graphmodel = (composer, new_model)
                # Also update it in the environment's internal store.
                env.agents_graphmodels[agent_id] = (composer, new_model)
        
        # Each agent trains/evaluates on the newly generated dataset.
        accuracies = {}
        for agent_id in agents:
            # Get the agent's graphmodel from the state.
            _, model = state.agents_states[agent_id].graphmodel
            acc = agents[agent_id].evaluate_accuracy(model, env.dataset)
            accuracies_history[agent_id].append(acc)
            accuracies[agent_id] = acc
    
    return actions_history, rewards_history, accuracies_history
