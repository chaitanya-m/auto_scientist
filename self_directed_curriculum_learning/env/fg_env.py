import gymnasium as gym
from gymnasium import spaces
import numpy as np
import uuid
import random

from curriculum_generator.problems import Problem
from env.utils.nn import create_minimal_graphmodel
from env.graph.composer import GraphTransformer
from env.graph.node import SingleNeuron, SubGraphNode
from env.utils.graph_utils import compute_complexity

class FunctionGraphEnv(gym.Env):
    """
    A Gymnasium environment for neural‐architecture search on any Problem.

    Observation:
      3D float vector: [current_mse, num_nodes, num_actions]

    Actions (Discrete(3)):
      0 = add_neuron
      1 = delete_repository_entry
      2 = add_from_repository

    valid_actions():
      returns subset of {0,1,2} — 2 only if repository non-empty.

    Reward:
      (reference_mse / reference_complexity) - (candidate_mse / candidate_complexity)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 problem: Problem,
                 train_epochs: int = 5,
                 batch_size: int = 100,
                 seed: int = 0):
        super().__init__()
        assert isinstance(problem, Problem), "problem must implement the Problem interface"
        self.problem = problem
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        # ground‐truth scalars
        self.reference_mse = problem.reference_mse
        self.reference_complexity = problem.reference_complexity()

        # build minimal graph using declared dims
        self.input_dim = problem.input_dim
        self.latent_dim = problem.output_dim

        self.action_space = spaces.Discrete(3)
        low = np.array([0.0, 1.0, 0.0], dtype=float)
        high = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)

        random.seed(seed)
        np.random.seed(seed)
        # NEW: Initialize repository here so it persists across resets/problems.
        self.repository = []
        self.reset()

    def reset(self, *, seed=None, options=None):
        """
        Reset to an empty graph while keeping the persistent repository.
        """
        composer, _ = create_minimal_graphmodel(
            (self.input_dim,),
            output_units=self.latent_dim,
            activation="relu"
        )
        self.composer = composer
        # Remove reinitialization of repository:
        # self.repository = []   <-- REMOVED to persist repository between problems
        self.best_mse = float('inf')
        self.graph_actions = []
        self.deletion_count = 0
        self.improvement_count = 0
        self.current_mse = 1.0

        return self._get_obs(), {}

    def valid_actions(self):
        """
        Returns valid action indices. 2 only if repository non-empty.
        """
        valids = [0, 1]
        if self.repository:
            valids.append(2)
        return valids

    def step(self, action):
        """
        Apply action, retrain briefly, update repository if improved,
        and return (obs, reward, done, truncated, info).
        """
        assert action in self.valid_actions(), f"Invalid action {action}"
        act_map = {0: "add_neuron", 1: "delete_repository_entry", 2: "add_from_repository"}
        act = act_map[action]
        self.graph_actions.append(act)

        if act == "add_neuron":
            node = SingleNeuron(name=str(uuid.uuid4()), activation="relu")
            try:
                self.composer.disconnect("input", "output")
            except Exception:
                pass
            self.composer.add_node(node)
            self.composer.connect("input", node.name)
            self.composer.connect(node.name, "output")

        elif act == "delete_repository_entry":
            if self.repository:
                idx = random.randrange(len(self.repository))
                del self.repository[idx]
                self.deletion_count += 1

        else:  # add_from_repository
            best_entry = max(self.repository, key=lambda e: e["utility"])
            transformer = GraphTransformer(self.composer)
            transformer.add_abstraction_node(
                abstraction_node=best_entry["subgraph_node"],
                chosen_subset=["input"],
                outputs=["output"],
                remove_prob=1.0
            )

        # draw data from the problem
        (Xtr, Ytr), (Xte, Yte) = self.problem.sample_batch(self.batch_size)
        # ground truth outputs
        ytr = self.problem.reference_output(Xtr)
        yte = self.problem.reference_output(Xte)

        # train candidate to match reference
        model = self.composer.build()
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                            epochs=self.train_epochs, verbose=0)
        mse = history.history["val_loss"][-1]
        self.current_mse = mse

        if mse < self.best_mse:
            self.best_mse = mse
            self.improvement_count += 1
            sub = SubGraphNode(name=f"sub_{uuid.uuid4().hex}", model=self.composer.build())
            transformer = GraphTransformer(self.composer)
            transformer.add_abstraction_node(
                abstraction_node=sub,
                chosen_subset=["input"],
                outputs=["output"],
                remove_prob=1.0
            )
            self.repository.append({"subgraph_node": sub, "utility": -mse})

        cand_complexity = compute_complexity(self.composer)
        reward = (self.reference_mse / self.reference_complexity) - (mse / cand_complexity)

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        """
        Returns the current observation: [current_mse, number of nodes, number of actions].
        """
        return np.array([
            self.current_mse,
            len(self.composer.nodes),
            len(self.graph_actions)
        ], dtype=float)

    def render(self):
        """
        Simple textual representation of the current state.
        """
        print(f"MSE={self.current_mse:.4f} | Nodes={len(self.composer.nodes)} "
              f"| Actions={len(self.graph_actions)} | Repo={len(self.repository)}")
