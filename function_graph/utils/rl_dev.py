import numpy as np

# Environment providing state and valid actions at each step.
class RLEnvironment:
    def __init__(self, total_steps=1000):
        self.total_steps = total_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # The state here is simply the current step number.
        return {"step": self.current_step}

    def valid_actions(self):
        # Two valid actions: "reuse" or "new"
        return ["reuse", "new"]

    def step(self, action):
        # In a full implementation, the action would affect performance.
        self.current_step += 1
        next_state = self._get_state()
        reward = 0  # Dummy reward; our test focuses on action frequency.
        done = self.current_step >= self.total_steps
        return next_state, reward, done

# Dummy agent whose probability of choosing "reuse" increases with each step.
class DummyAgent:
    def __init__(self, total_steps=1000):
        self.total_steps = total_steps
        self.actions_history = []

    def choose_action(self, state, valid_actions):
        step = state["step"]
        # Reuse probability increases linearly with the step number.
        reuse_prob = min(1.0, step / self.total_steps)
        action = "reuse" if np.random.rand() < reuse_prob else "new"
        self.actions_history.append(action)
        return action

# Run one episode by stepping through the environment.
def run_episode(env, agent):
    state = env.reset()
    while True:
        actions = env.valid_actions()
        action = agent.choose_action(state, actions)
        state, reward, done = env.step(action)
        if done:
            break
    return agent.actions_history

# Example execution when running the file directly.
if __name__ == "__main__":
    env = RLEnvironment(total_steps=1000)
    agent = DummyAgent(total_steps=1000)
    actions = run_episode(env, agent)
    print(f"Reuse count: {actions.count('reuse')}, New count: {actions.count('new')}")
