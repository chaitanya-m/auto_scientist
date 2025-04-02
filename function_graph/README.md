
# An attempt at self-determined curriculum learning: getting an agent to learn a hard concept by requesting easier concepts to learn, and building a repository of motifs or patterns that can then be used to solve the harder problem, starting from basic functions. While Occam's Razor may be implicit in the feedback when a single agent is compared against a reference in terms of accuracy/complexity, allowing competing agents should make redundant the need to do so as agents will have to maintain their motif repositories at a finite, naturally bounded level in order to stay competitive.

<!-- 

## Round:

A single interaction cycle where agents receive data (generated from a fixed distribution using a specific seed), build/modify their function graph, classify the data, and receive rewards.
## Episode:

A series of rounds that use data generated from the same distribution and seed. At the end of an episode, performance is evaluated, and the environment can adjust internal parameters.

## Experiment:
A collection of episodes. In some experiments, the data distribution remains constant between episodes; in others, it varies between episodes to test robustness or adaptability.

# Reinforcement Learning Strategies

---

## Multi-Armed Bandit
In a multi-armed bandit problem, an agent repeatedly selects one of several possible actions (or “arms”) to maximize its expected reward. There are no state transitions here—the focus is solely on balancing exploration (trying different arms) with exploitation (choosing the arm with the highest expected reward).

---

## Contextual Bandit
A contextual bandit extends the multi-armed bandit by incorporating context (or state) information. The agent observes the context and selects an action that maximizes the expected reward given that context. Although context is used, the decision is still made without considering long-term state transitions, that is, future states are not considered.

---

## Q-Learning
In Q-Learning, we are interested in the expectation of reward for state–action pairs; this allows planning into the future - as a policy maps each state-action pair to an expected reward, we can choose to follow a policy that maximizes reward expectation.

We update it with the following formula:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr]
$$

Here’s what happens:
- **Reward Term:** The immediate reward $ r $ is observed.
- **Future Expectation:** We add the discounted maximum expected reward from the next state, $ \gamma \max_{a'} Q(s',a') $.
- **Baseline Correction:** We subtract the current expectation $ Q(s,a) $ to measure the difference (or error) between the current belief and the new information.
- **Update:** We adjust $ Q(s,a) $ by a fraction $ \alpha $ of that difference.

This update is off-policy because it uses the maximum over future actions—i.e., it assumes the agent will act optimally in the future, regardless of the current (possibly exploratory) policy.

---

## SARSA (State-Action-Reward-State-Action)
In SARSA, we also estimate the expectation of reward for state–action pairs, but the update is based on the action actually taken in the next state. The update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl[\, r + \gamma Q(s',a') - Q(s,a)\Bigr]
$$

Key points:
- **Actual Action:** $ a' $ is the action chosen by the current policy.
- **Baseline Correction:** Again, we subtract $ Q(s,a) $ as a baseline.
- **On-Policy Update:** This update uses the expectation of the reward according to the current policy, making SARSA an on-policy algorithm.

---

## Monte Carlo Methods
Monte Carlo methods learn the expectation of reward by averaging the total rewards (returns) received over complete episodes. 

Instead of updating Q-values in the table at each step, they wait until an episode finishes and then use the observed return as an unbiased estimate of the expected reward.

In Monte Carlo methods, after an episode finishes, we "backtrack" through the episode to update Q-values. For each state–action pair $(s_t,a_t)$ encountered at time $t$, we compute the return as a discounted series of rewards from there onwards:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T
$$

Then, we update the Q-value using return from time $t$, $G_t$:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \Bigl[ G_t - Q(s_t,a_t) \Bigr]
$$

This update rule adjusts the expectation of reward for $(s_t,a_t)$ by moving it toward the return $G_t$ from the present episode, using the difference as the error term.


---

## Policy Gradient Methods
Policy gradient methods directly adjust the parameters of the policy to maximize the expected reward. Rather than estimating a value for each state–action pair, these methods compute the gradient of the expected return with respect to the policy parameters and update the policy in the direction that increases that expectation.

---

## Actor-Critic Methods
Actor-Critic methods combine the benefits of value-based and policy-based approaches:
- **Actor:** Directly updates the policy.
- **Critic:** Estimates the expected reward (or advantage) to serve as a baseline, reducing the variance in the policy gradient updates.
This dual structure helps improve the learning process.

---

## Deep Q-Networks (DQN)
DQN extends Q-Learning by using deep neural networks to approximate the Q-value function. This allows the agent to handle high-dimensional state spaces. The update rule remains the same as in Q-Learning, but the Q-values are now produced by a neural network, which learns to estimate the expected reward over time.

---

## Proximal Policy Optimization (PPO)
PPO is a policy gradient method that maximizes the expected reward while ensuring that the policy updates do not change the policy too much. It uses a surrogate objective function with a clipping mechanism to maintain stability during training.

---

## Asynchronous Advantage Actor-Critic (A3C)
A3C runs multiple agents in parallel, each with its own copy of the environment. It uses an actor-critic structure, where the critic estimates the advantage (the extra reward over the expected reward) to guide the actor's updates. This parallelism helps stabilize and speed up learning in complex environments.

---

## Self-Play
In self-play, an agent learns by competing against copies of itself. By continuously playing against increasingly strong versions of itself, the agent improves its strategy to maximize the expected reward. This method has been especially effective in competitive games like chess or Go.

---









# Algorithm for Building a Function Graph with Gradual Complexity Increase and Adaptive Node Manipulation

## Initialization

1. Create an empty graph  *G*  with an input node  *I<sub>in</sub>*  and an output node  *O<sub>out</sub>*.
2. Initialize the component basket  *C*  with only the sigmoid node  σ.
3. Set the initial maximum number of components to  *K* = 1.
4. Set the maximum allowed value for  *K*  to  *K<sub>max</sub>*.

## Graph Construction and Evaluation

1. **Iteration:** While  *K* ≤ *K<sub>max</sub>*:
    - **Add Components:**
        - For  *i* = 1 to  *K*:
            - **Select Component:** Select a component  *c<sub>i</sub>*  from the basket  *C*.
            - **Select Subset:** Select a subset  *S*  of existing nodes in  *G*  to connect  *c<sub>i</sub>* to.
                - Initially,  *S*  can only be {*I<sub>in</sub>*}.
                - In subsequent iterations,  *S*  can be any subset of existing nodes, including  *I<sub>in</sub>*.
            - **Add and Connect:** Add  *c<sub>i</sub>*  to  *G*  and connect it to the nodes in  *S*, ensuring shape compatibility using adapter creation, padding, or input subset selection/multiple copies.
    - **Connect to Output:** Select a subset  *O*  of nodes in  *G*  (excluding  *I<sub>in</sub>*) and connect them to the output node  *O<sub>out</sub>*.
    - **Build Neural Network:** Build a neural network  *N*  from the graph  *G*, where each node represents a neural network component (e.g.,  σ).
    - **Train and Test:** Train  *N*  on the training data and evaluate its performance  *P*  on the test data.
    - **Feedback:** Based on  *P*, decide whether to:
        - **Increase Complexity:** Increase  *K*  by 1 (if  *K* < *K<sub>max</sub>*) and add more components using **Add Node**.
        - **Decrease Complexity:** Delete an existing component from  *G*  using **Delete Node**.
        - **Modify Connections:** Change the subset  *O*  of nodes connected to  *O<sub>out</sub>*  using **Change Output Connections**.
        - **Modify Components:** Replace an existing component in  *G*  with a different one from  *C*.
2. **Select Best:** Select the graph  *G*  with the best performance  *P*  among all graphs evaluated.

## Compression and Basket Update

1. **Compression:** If  *P*  of the selected graph meets the desired criteria:
    - Compress  *G*  into a reusable neural network module  *M*.
    - Add  *M*  to the component basket  *C*.
2. **Basket Management:** If the size of  *C*  exceeds a predefined limit, remove the least used or least performing components to maintain a manageable compositional space.

## Observation of Emergent Structure

1. Analyze the structure of the final graph  *G*  to determine if it resembles a layered neural network, even though the search process did not explicitly enforce this constraint.

## Node Manipulation Functions

- **Add Node (*c*,  *S*):** Adds a new component  *c*  from  *C*  to  *G*, connecting it to the nodes in subset  *S*.
- **Delete Node (*c*):** Removes an existing component  *c*  from  *G*, adjusting connections as needed.
- **Change Output Connections (*O*):** Updates the subset  *O*  of nodes connected to the output node  *O<sub>out</sub>*.



















# Draft Algorithm for Building a Function Graph with MCTS, Sigmoid Nodes, and Compression

## Initialization

1. Create an empty graph  *G*.
2. Add a single sigmoid node  *s*  to  *G*  and connect it to the input.
3. Initialize the MCTS tree with  *G*  as the root node.
4. Initialize the component basket  *C*  with only the sigmoid node.

## MCTS Iteration

1. **Selection:** Starting from the root node, traverse the tree using the UCB1 selection policy until a leaf node  *L*  is reached.
2. **Expansion:** If  *L*  is not a terminal node (graph size limit not reached, performance not satisfactory), expand it by adding a new child node for each possible action:
    - Add a new node  *n*  from the basket  *C*  to the graph.
    - Connect  *n*  to existing nodes in the graph (including the input) or subsets of nodes, ensuring shape compatibility using adapter creation, padding, or input subset selection/multiple copies.
3. **Simulation:** For each new child node, simulate a rollout by randomly adding nodes from  *C*  and connections until a terminal state is reached. Evaluate the performance of the resulting graph and assign a reward.
4. **Backpropagation:** Backpropagate the reward up the tree, updating the visit counts and average rewards of the visited nodes.
5. **Repeat:** Repeat steps 1-4 for a fixed number of iterations or until a satisfactory graph is found.

## Termination and Compression

1. Select the child node of the root with the highest visit count as the best action.
2. Apply the corresponding action to the current graph.
3. **Compression:** If the performance of the resulting graph meets the desired criteria:
    - Compress the graph into a reusable neural network module  *M*.
    - Add  *M*  to the component basket  *C*.
4. **Termination:** If the performance meets the desired criteria or a maximum number of iterations is reached, terminate the algorithm and return the best graph found.
5. Otherwise, continue with the MCTS iteration.

## Observation of Emergent Structure

1. Analyze the structure of the final graph to determine if it resembles a layered neural network, even though the search process did not explicitly enforce this constraint.

## Notes

- The constraint on graph size can be defined based on the input and output shapes and the complexity of the task.
- The reward function should encourage the creation of graphs that achieve good performance on the task.
- The UCB1 selection policy balances exploration and exploitation to efficiently search the space of possible graph structures.
- The simulation phase can be optimized by using heuristics or domain knowledge to guide the random node additions and connections.
- The analysis of the emergent structure can involve visualizing the graph, calculating metrics related to layering (e.g., number of layers, connectivity patterns), and comparing the graph to known neural network architectures.
- Compression of successful architectures into reusable modules allows for the reuse of learned structures and potentially accelerates the search process in later iterations.









# Algorithm for Building a Function Graph with MCTS, Sigmoid Nodes, and Compression

## Initialization

1. Create an empty graph `G`.
2. Add a single sigmoid node `s` to `G` and connect it to the input.
3. Initialize the MCTS tree with `G` as the root node.
4. Initialize the component basket `C` with only the sigmoid node.

## MCTS Iteration

1. **Selection:** Starting from the root node, traverse the tree using the UCB1 selection policy until a leaf node `L` is reached.
2. **Expansion:** If `L` is not a terminal node (graph size limit not reached, performance not satisfactory), expand it by adding a new child node for each possible action:
    - Add a new node `n` from the basket `C` to the graph.
    - Connect `n` to existing nodes in the graph (including the input) or subsets of nodes, ensuring shape compatibility using adapter creation, padding, or input subset selection/multiple copies.
3. **Simulation:** For each new child node, simulate a rollout by randomly adding nodes from `C` and connections until a terminal state is reached. Evaluate the performance of the resulting graph and assign a reward.
4. **Backpropagation:** Backpropagate the reward up the tree, updating the visit counts and average rewards of the visited nodes.
5. **Repeat:** Repeat steps 1-4 for a fixed number of iterations or until a satisfactory graph is found.

## Termination and Compression

1. Select the child node of the root with the highest visit count as the best action.
2. Apply the corresponding action to the current graph.
3. **Compression:** If the performance of the resulting graph meets the desired criteria:
    - Compress the graph into a reusable neural network module `M`.
    - Add `M` to the component basket `C`.
4. **Termination:** If the performance meets the desired criteria or a maximum number of iterations is reached, terminate the algorithm and return the best graph found.
5. Otherwise, continue with the MCTS iteration.

## Observation of Emergent Structure

1. Analyze the structure of the final graph to determine if it resembles a layered neural network, even though the search process did not explicitly enforce this constraint.

## Notes

- The constraint on graph size can be defined based on the input and output shapes and the complexity of the task.
- The reward function should encourage the creation of graphs that achieve good performance on the task.
- The UCB1 selection policy balances exploration and exploitation to efficiently search the space of possible graph structures.
- The simulation phase can be optimized by using heuristics or domain knowledge to guide the random node additions and connections.
- The analysis of the emergent structure can involve visualizing the graph, calculating metrics related to layering (e.g., number of layers, connectivity patterns), and comparing the graph to known neural network architectures.
- Compression of successful architectures into reusable modules allows for the reuse of learned structures and potentially accelerates the search process in later iterations.










# MCTS-based Predictor Building: Avoiding Shape Mismatches

## Environment

- **Input:** A set of input features denoted by $X = \{x_1, x_2, ..., x_n\}$, where $x_i$ represents the $i$-th input feature.
- **Output:** A target variable denoted by $Y$.
- **Component Basket:** A set of reusable components (e.g., ReLU, Sigmoid) denoted by $C = \{c_1, c_2, ..., c_m\}$, where $c_j$ represents the $j$-th component.
- **Baseline Model:** A pre-trained model with satisfactory performance on the dataset, used as a target for the Agents.

## Agent Actions

Agents can perform the following actions to modify their predictor graph:

1. **Add Node:**
    - Select a component $c_j$ from the basket $C$.
    - Connect the component to existing nodes in the graph, ensuring shape compatibility using the following strategies:
        - **Adapter Creation:** If the output shape of the source node $s$ does not match the input shape of the target component $c_j$, create an adapter $a$ that transforms the output of $s$ to match the input shape of $c_j$. The adapter is a trainable layer that learns the necessary transformation.
        - **Component Selection:** If a component $c_k$ in the basket $C$ has an output shape that matches the input shape of $c_j$, select $c_k$ as the source node instead of creating an adapter.
        - **Input Padding:** If the input shape of $c_j$ is larger than the output shape of the source node $s$, pad the output of $s$ with zeros or duplicate inputs from other nodes until the input shape of $c_j$ is satisfied. This can be represented as $s' = pad(s, shape(c_j))$, where $s'$ is the padded output of $s$.
        - **Input Subset Selection/Multiple Copies:** If the input shape of $c_j$ is smaller than the output shape of the source node $s$, either select a subset of the outputs of $s$ that matches the input shape of $c_j$, or create multiple copies of $c_j$ and distribute the outputs of $s$ among them. This can be represented as $s' = subset(s, shape(c_j))$ or $c_j' = replicate(c_j, shape(s))$, where $s'$ is the selected subset of outputs and $c_j'$ is the set of replicated components.

2. **Delete Node:**
    - Remove a node from the graph, ensuring that the remaining nodes still have valid connections.
    - If deleting a node creates a shape mismatch, apply the strategies described in "Add Node" to restore shape compatibility.

3. **Modify Node:**
    - Change the properties of an existing node (e.g., activation function), ensuring that the modification does not introduce shape mismatches.
    - If a modification creates a shape mismatch, apply the strategies described in "Add Node" to restore shape compatibility.

## MCTS Algorithm

The MCTS algorithm is used to guide the Agent's actions in building the predictor graph. The algorithm follows the standard MCTS steps: Selection, Expansion, Simulation, and Backpropagation.

## Objective

The Agents compete to build a predictor graph that reaches the baseline performance first. The Agent that achieves this objective wins the game.

## Mathematical Notation Summary

- $X$: Set of input features.
- $Y$: Target variable.
- $C$: Set of reusable components.
- $c_j$: The $j$-th component in the basket.
- $s$: Source node.
- $a$: Adapter.
- $pad(s, shape(c_j))$: Padding operation to match the input shape of $c_j$.
- $subset(s, shape(c_j))$: Subset selection operation to match the input shape of $c_j$.
- $replicate(c_j, shape(s))$: Replication operation to handle all outputs of $s$.

 -->



