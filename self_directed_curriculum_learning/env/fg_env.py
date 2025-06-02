import gymnasium as gym
from gymnasium import spaces
import numpy as np
import uuid
import random

from curriculum_generator.curriculum_interface import CurriculumInterface
from env.utils.nn import create_minimal_graphmodel
from env.graph.composer import GraphTransformer
from env.graph.node import SingleNeuron, SubGraphNode
from env.utils.graph_utils import compute_complexity


class FunctionGraphEnv(gym.Env):
    """
    A Gymnasium environment for neural-architecture search on any Problem.

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

    def __init__(
        self,
        *,
        curriculum: CurriculumInterface,
        train_epochs: int = 5,
        batch_size: int = 100,
        seed: int = 0,
        config: dict = None  # Configuration dictionary for scaling factors
    ):
        super().__init__()
        # now driven by a Curriculum rather than a single Problem
        assert isinstance(curriculum, CurriculumInterface), \
            "curriculum must implement CurriculumInterface"
        self.curriculum = curriculum
        self._problem_iter = iter(curriculum)

        self.train_epochs = train_epochs
        self.batch_size = batch_size

        # Default configuration for difficulty adjustment rewards
        self.config = config or {
            "increase_scale": 0.5,  # 50% of episode reward for harder problems
            "decrease_scale": -0.5,  # -50% of episode reward for easier problems
            "maintain_scale": 0.0   # No reward for maintaining difficulty
        }

        # ground-truth scalars (set per-problem in reset)
        self.reference_mse = None
        self.reference_complexity = None

        # dims (set per-problem in reset)
        self.input_dim = None
        self.latent_dim = None

        # action & observation spaces (0–2 = graph ops; 3=↑diff, 4=↓diff, 5=maintain)
        self.action_space = spaces.Discrete(6)
        low = np.array([0.0, 1.0, 0.0], dtype=float)
        high = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)

        random.seed(seed)
        np.random.seed(seed)

        # persistent repository of learned abstractions
        self.repository = []

        # delay pulling the first problem until reset() is called
        self.current_problem = None
        self.composer = None

        # phase flag: after a graph step+reward, await difficulty action
        self._awaiting_diff = False

        # initialize episode-specific fields so they exist prior to reset()
        self.best_mse = float('inf')
        self.graph_actions = []
        self.deletion_count = 0
        self.improvement_count = 0
        self.current_mse = 1.0

        # Track the best observation vector during the episode
        self.best_obs = None

    def reset(self, *, seed=None, options=None):
        """
        Reset to an empty graph while keeping the persistent repository.
        Pulls the next Problem from the curriculum.
        """
        # 1) Pull next problem
        self.current_problem = next(self._problem_iter)

        # 2) Update per-problem scalars
        self.reference_mse = self.current_problem.reference_mse
        self.reference_complexity = self.current_problem.reference_complexity()

        # 3) Update dims
        self.input_dim = self.current_problem.input_dim
        self.latent_dim = self.current_problem.output_dim

        # 4) Build minimal starter graph
        composer, _ = create_minimal_graphmodel(
            (self.input_dim,),
            output_units=self.latent_dim,
            activation="relu"
        )
        self.composer = composer

        # Episode-specific counters
        self.best_mse = float('inf')
        self.graph_actions = []
        self.deletion_count = 0
        self.improvement_count = 0
        self.current_mse = 1.0

        # fresh problem → next step is a graph step
        self._awaiting_diff = False

        return self._get_obs(), {}

    def valid_actions(self):
        """
        Two‐phase:
         - If awaiting a diff‐change (and the curriculum implements all three),
           only [3,4,5] are valid.
         - Otherwise standard graph ops [0,1,(2)].
        """
        # phase 2: difficulty choices only if hooks are actually provided
        inc_fn = getattr(self.curriculum, "_increase_fn", None)
        dec_fn = getattr(self.curriculum, "_decrease_fn", None)
        keep_fn = getattr(self.curriculum, "_maintain_fn", None)
        if self._awaiting_diff and inc_fn and dec_fn and keep_fn:
            return [3, 4, 5]

        # phase 1: graph ops
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

        # phase 1: apply a graph operation
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

        # draw data from the current problem
        (Xtr, Ytr), (Xte, Yte) = self.current_problem.sample_batch(self.batch_size)
        ytr = self.current_problem.reference_output(Xtr)
        yte = self.current_problem.reference_output(Xte)

        # train candidate to match reference
        model = self.composer.build()
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                            epochs=self.train_epochs, verbose=0)
        mse = history.history["val_loss"][-1]
        self.current_mse = mse

        # Update the best observation vector if the current MSE improves
        if mse < self.best_mse:
            self.best_mse = mse
            self.improvement_count += 1
            self.best_obs = self._get_obs()  # Save the best observation vector
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

        # include current difficulty so your logs can see it
        info = {"difficulty": self.curriculum.difficulty}
        return self._get_obs(), reward, False, False, info

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
        print(
            f"MSE={self.current_mse:.4f} | "
            f"Nodes={len(self.composer.nodes)} | "
            f"Actions={len(self.graph_actions)} | "
            f"Repo={len(self.repository)} | "
            f"Difficulty={self.curriculum.difficulty}"
        )

    def clone(self):
        """
        Create a deep copy of this environment at its current episode state, including:
          - A cloned GraphComposer (architecture + weights)
          - Copied scalar state and history
          - A fresh list for the repository (entries shallow-copied)
          - A one-shot iterator that will re-play the same current_problem

        We do *not* deep‐clone each repository entry; they’ll be cloned on demand
        by GraphTransformer.add_abstraction_node when used.
        """
        # 1) New env shell with identical configuration
        new_env = FunctionGraphEnv(
            curriculum=self.curriculum,
            train_epochs=self.train_epochs,
            batch_size=self.batch_size,
            seed=None  # RNG can be reseeded separately if needed
        )

        # 2) Copy over per-problem scalars & dims
        new_env.current_problem     = self.current_problem
        new_env.reference_mse        = self.reference_mse
        new_env.reference_complexity = self.reference_complexity
        new_env.input_dim            = self.input_dim
        new_env.latent_dim           = self.latent_dim

        # 3) Clone the composer (weights + topology)
        new_env.composer = self.composer.clone()

        # 4) Shallow-copy repository list only
        new_env.repository = list(self.repository)

        # 5) Copy episode-specific counters & history
        new_env.best_mse          = self.best_mse
        new_env.graph_actions     = list(self.graph_actions)
        new_env.deletion_count    = self.deletion_count
        new_env.improvement_count = self.improvement_count
        new_env.current_mse       = self.current_mse
        # *** copy the phase flag so valid_actions() stays in sync ***
        new_env._awaiting_diff    = self._awaiting_diff

        return new_env

    def adjust_difficulty(self, action, episode_reward):
        """
        Adjust the difficulty of the curriculum after the episode ends.
        This method should be called explicitly after all cloned environments are closed.
        Assigns a reward based on the difficulty adjustment action and the episode reward.
        
        Args:
            action (int): Difficulty adjustment action (3=↑diff, 4=↓diff, 5=maintain).
            episode_reward (float): Total reward earned during the episode.
        """
        assert action in [3, 4, 5], f"Invalid difficulty adjustment action {action}"

        if action == 3:
            self.curriculum.increase_difficulty()
            act = "increase_difficulty"
            reward = self.config["increase_scale"] * episode_reward
        elif action == 4:
            self.curriculum.decrease_difficulty()
            act = "decrease_difficulty"
            reward = self.config["decrease_scale"] * episode_reward
        else:
            self.curriculum.maintain_difficulty()
            act = "maintain_difficulty"
            reward = self.config["maintain_scale"] * episode_reward

        return {"action": act, "difficulty": self.curriculum.difficulty, "reward": reward}
