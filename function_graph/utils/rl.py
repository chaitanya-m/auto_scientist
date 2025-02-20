# utils/rl.py
from graph.composer import GraphTransformer


def run_episode(env, agents, seed=0, steps = 0, schema=None, step_callback=None):
    """
    Runs an episode of the environment for exactly env.total_steps steps,
    for one or more agents. Each agent can choose actions (including
    "add_abstraction") at each step.

    Parameters
    ----------
    env : RLEnvironment
        The environment instance, which must contain a repository
        with any learned abstraction(s) for "add_abstraction" to work.
    agents : dict[int, AgentInterface]
        A dictionary mapping agent IDs to agent instances that have
        choose_action(...), evaluate_accuracy(...), etc.
    seed : int, optional
        Seed for resetting the environment, by default 0.
    schema : optional
        Optional new schema for environment reset, by default None.
    step_callback : callable, optional
        A function that is invoked at each step before agents choose actions.
        Its signature should be: step_callback(current_step, state, agents, env)
        - current_step (int): The current step index in the environment.
        - state (State): The current State object from env.
        - agents (dict[int, AgentInterface]): The mapping of agent IDs to agent instances.
        - env (RLEnvironment): The environment instance.
        This callback can be used to update valid actions, change agent policies,
        or log data at each step.

    Returns
    -------
    actions_history : dict[int, list[str]]
        The sequence of actions each agent took, keyed by agent ID.
    rewards_history : dict[int, list[float]]
        The sequence of rewards each agent received over env.total_steps, keyed by agent ID.
    accuracies_history : dict[int, list[float]]
        The sequence of accuracies each agent achieved at each step, keyed by agent ID.
    """
    # 1. Reset environment to a fresh state. Note that the returned state is a State instance.
    state = env.reset(seed=seed, new_schema=schema)
    
    # 2. Compile models for each agent if needed.
    for agent_id, (composer, model) in env.agents_graphmodels.items():
        if not hasattr(model, "optimizer"):
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Prepare history dictionaries.
    actions_history = {agent_id: [] for agent_id in env.agents_graphmodels}
    rewards_history = {agent_id: [] for agent_id in env.agents_graphmodels}
    accuracies_history = {agent_id: [] for agent_id in env.agents_graphmodels}
    debug_counter = {agent_id: 0 for agent_id in env.agents_graphmodels}
    
    # 3. Main loop over total_steps.
    for step_count in range(steps):
        # Generate new data and update the environment state.
        state = env.step()

        # Optional: give the test code a chance to modify agents, valid actions, etc.
        if step_callback is not None:
            step_callback(env.current_step, state, agents, env)

        # Each agent picks an action based on its AgentState from the overall State object.
        chosen_actions = {}
        for agent_id in agents:
            # Use the AgentState from the new State object.
            agent_state = state.agents_states[agent_id]
            action = agents[agent_id].choose_action(agent_state, step_count)
            chosen_actions[agent_id] = action
            actions_history[agent_id].append(action)

        # 4. Apply any "add_abstraction" actions.
        for agent_id, action in chosen_actions.items():
            if action == "add_abstraction":
                debug_counter[agent_id] += 1
                print(f"DEBUG: {agent_id} adds abstraction {debug_counter[agent_id]} times")
                learned_abstraction = env.repository["learned_abstraction"]
                composer, model = env.agents_graphmodels[agent_id]
                transformer = GraphTransformer(composer)
                
                # For simplicity, always connect "input" -> abstraction -> "output", removing direct input->output.
                new_model = transformer.add_abstraction_node(
                    abstraction_node=learned_abstraction,
                    chosen_subset=["input"],
                    outputs=["output"],
                    remove_prob=1.0
                )
                env.agents_graphmodels[agent_id] = (composer, new_model)

        # 5. Each agent trains/evaluates on the newly generated dataset.
        accuracies = {}
        for agent_id in agents:
            _, model = env.agents_graphmodels[agent_id]
            acc = agents[agent_id].evaluate_accuracy(model, env.dataset)
            accuracies_history[agent_id].append(acc)
            accuracies[agent_id] = acc

        # 6. Compute rewards using the environment's logic.
        rewards = env.compute_rewards(accuracies)
        for agent_id in agents:
            rewards_history[agent_id].append(rewards[agent_id])

    return actions_history, rewards_history, accuracies_history

