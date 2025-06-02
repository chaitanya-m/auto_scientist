# function_graph/agents/mcts.py

import math
import random
import numpy as np

class SimulationStepPolicyNetwork:
    """
    Simple MLP policy network mapping state feature vectors to action probabilities.
    Input: raw observation vector (env.observation_space.shape)
    Output: probability over env.action_space.n actions
    """
    def __init__(self, model):
        """
        Args:
            model (tf.keras.Model): A compiled Keras model with softmax output.
        """
        self.model = model

    def predict(self, obs):
        """
        Predict action probabilities given an observation.

        Args:
            obs (np.array): Observation vector of shape (obs_dim,).

        Returns:
            np.array: Probability vector over actions of shape (action_dim,).
        """
        return self.model.predict(np.atleast_2d(obs), verbose=0)[0]


class EpisodePolicyNetwork:
    """
    Policy network for difficulty adjustment actions.
    Maps state observations and episode reward to probabilities over difficulty actions.
    """
    def __init__(self, model):
        """
        Args:
            model (tf.keras.Model): A compiled Keras model with softmax output.
        """
        self.model = model

    def predict(self, obs, episode_reward):
        """
        Predict difficulty adjustment probabilities given an observation and episode reward.

        Args:
            obs (np.array): Observation vector of shape (obs_dim,).
            episode_reward (float): Total reward earned during the episode.

        Returns:
            np.array: Probability vector over difficulty actions [increase, decrease, maintain].
        """
        input_data = np.concatenate([obs, [episode_reward]])
        return self.model.predict(np.atleast_2d(input_data), verbose=0)[0]


class MCTSNode:
    """
    Node in an MCTS search tree.
    """
    def __init__(self, obs, parent=None, action=None):
        self.obs = obs
        self.parent = parent
        self.action = action
        self.children = {}     # action_idx -> MCTSNode
        self.visits = 0
        self.total_value = 0.0

    def average_value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0


class SimpleMCTSAgent:
    """
    Monte Carlo Tree Search agent for Gymnasium-style environments.
    Expects the environment to implement:
      - reset(): returns (obs, info)
      - clone(): returns a deep copy of the env at its current state
      - valid_actions(): returns list of valid action indices
      - step(action): returns (obs, reward, done, truncated, info)
      - observation_space, action_space
    """
    def __init__(self, env, policy_model, difficulty_policy_model, search_budget=20, c=1.41):
        """
        Args:
            env: The environment instance.
            policy_model: The policy network for MCTS action selection.
            difficulty_policy_model: The policy network for difficulty adjustment.
            search_budget: Number of simulations per MCTS search.
            c: Exploration constant for UCB.
        """
        self.env = env
        self.policy = SimulationStepPolicyNetwork(policy_model)  # Existing policy network for MCTS actions
        self.difficulty_policy = EpisodePolicyNetwork(difficulty_policy_model)  # New policy network for difficulty adjustment
        self.search_budget = search_budget
        self.c = c

    def mcts_search(self):
        """
        Perform MCTS for self.search_budget iterations.
        Returns the best action index to take from the root state.
        """
        # 1) Start from the current env state (problem already set)
        root_obs = self.env._get_obs()  # Use self.env instead of self.base_env
        self.root = MCTSNode(root_obs)

        # 2) Perform rollouts
        for _ in range(self.search_budget):
            leaf, sim_env = self._select(self.root)
            reward = self._simulate(leaf, sim_env)
            self._backpropagate(leaf, reward)

        # 3) Pick best child of root by average value
        if not self.root.children:
            # fallback: choose random valid action
            return random.choice(self.env.valid_actions())  # Use self.env instead of self.base_env
        best_child = max(self.root.children.values(), key=lambda n: n.average_value())
        return best_child.action

    def _select(self, node):
        """
        From `node`, clone the current environment, replay the path to `node`,
        then expand one child if none exist, otherwise descend by UCB.
        Returns (new_node, env_copy).
        """
        # Clone current env state (same problem, same graph, same repository)
        env = self.env.clone()  # Use self.env instead of self.base_env

        # Replay path from root to this node
        path = []
        cur = node
        while cur.parent:
            path.append(cur.action)
            cur = cur.parent
        for a in reversed(path):
            env.step(a)

        # Expansion if no children
        valid = env.valid_actions()
        if not node.children and valid:
            a = random.choice(valid)
            obs_next, _, _, _, _ = env.step(a)
            new_node = MCTSNode(obs_next, parent=node, action=a)
            node.children[a] = new_node
            return new_node, env

        # Selection by UCB
        while node.children:
            valid = env.valid_actions()
            probs = self.policy.predict(node.obs)
            total_visits = node.visits
            best_ucb, best_act, best_child = -float('inf'), None, None
            for a in valid:
                if a in node.children:
                    child = node.children[a]
                    exploit = child.average_value()
                    explore = self.c * math.sqrt(math.log(total_visits + 1) / (child.visits + 1))
                    ucb = (exploit + explore) * probs[a]
                else:
                    ucb = self.c * math.sqrt(math.log(total_visits + 1)) * probs[a]
                if ucb > best_ucb:
                    best_ucb, best_act = ucb, a
                    best_child = node.children.get(a)
            obs_next, _, _, _, _ = env.step(best_act)
            if best_child is None:
                new_node = MCTSNode(obs_next, parent=node, action=best_act)
                node.children[best_act] = new_node
                return new_node, env
            node = best_child

        return node, env

    def _simulate(self, node, env):
        """
        Perform a one-step rollout: pick a random valid action and return its reward.
        """
        legal = env.valid_actions()
        action = random.choice(legal)
        _, reward, _, _, _ = env.step(action)
        return reward

    def _backpropagate(self, node, reward):
        """
        Propagate reward up from `node` to root.
        """
        cur = node
        while cur:
            cur.visits += 1
            cur.total_value += reward
            cur = cur.parent

    def choose_difficulty_action(self, best_obs, episode_reward):
        """
        Use the difficulty policy network to choose a difficulty adjustment action.

        Args:
            best_obs (np.array): Best observation vector of shape (obs_dim,).
            episode_reward (float): Total reward earned during the episode.

        Returns:
            int: Difficulty adjustment action (3=↑diff, 4=↓diff, 5=maintain).
        """
        probs = self.difficulty_policy.predict(best_obs, episode_reward)
        return np.argmax(probs) + 3  # Map probabilities to actions [3, 4, 5]
