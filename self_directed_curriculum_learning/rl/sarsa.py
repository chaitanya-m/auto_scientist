"""sarsa.py

SARSA (State-Action-Reward-State-Action) implementation using the same 
interfaces as Q-learning. The key difference from Q-learning is that SARSA
is an **on-policy** algorithm:

* Q-Learning (off-policy): Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
  Uses the MAXIMUM Q-value for next state (assumes optimal future play)

* SARSA (on-policy): Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]  
  Uses the Q-value for the ACTUAL action taken in next state (accounts for exploration)

This makes SARSA more conservative because it learns about the policy it's 
actually following (including exploration), while Q-learning optimistically
assumes perfect play from the next state onward.

SARSA is often preferred in environments where exploration can be dangerous,
since it accounts for the fact that the agent will sometimes take suboptimal
actions due to exploration.
"""

from typing import Any, List, Mapping, Sequence, Union
from qlearning import create_qlearning_agent, QLearningAgent, TabularQ
from utils import evaluate_policy
from interfaces import ActionT, CriticUpdater, Transition


class SARSACriticUpdater(CriticUpdater[Any, ActionT]):
    """
    Updates TabularQ using SARSA (on-policy) updates.
    
    Key difference from Q-learning:
    - Q-learning: Uses max Q(s',a') for next state (off-policy, optimistic)
    - SARSA: Uses Q(s',a') for actual action taken (on-policy, conservative)
    
    SARSA accounts for the exploration policy when learning, making it more
    conservative but safer in environments where exploration has consequences.
    
    Example:
    Imagine state S with actions [LEFT=10, RIGHT=1], agent took action A, 
    got reward R=5, ended up in state S':
    
    Q-Learning: target = 5 + γ * max(10, 1) = 5 + γ * 10 (optimistic)
    SARSA: target = 5 + γ * Q(S', actual_action_taken) (realistic)
    
    If the agent explores and takes RIGHT due to ε-greedy:
    SARSA: target = 5 + γ * 1 (accounts for exploration)
    Q-Learning: target = 5 + γ * 10 (ignores exploration)
    """
    
    def __init__(self, q_table: TabularQ, alpha: float = 0.1, gamma: float = 0.99):
        self.q_table = q_table
        self.alpha = alpha
        self.gamma = gamma
    
    def loss(self, predictions: Sequence[float], targets: Sequence[float]) -> float:
        """Compute MSE loss for diagnostics."""
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    def step(self, experiences: Sequence[Transition[Any, ActionT]]) -> Mapping[str, float]:
        predictions = []
        targets = []
        
        for exp in experiences:
            pred = self.q_table.q(exp.state, exp.action)
            
            if exp.done:
                target = exp.reward
            else:
                # Use the actual next action from the transition
                if hasattr(exp, 'next_action') and exp.next_action is not None:
                    next_q = self.q_table.q(exp.next_state, exp.next_action)
                    target = exp.reward + self.gamma * next_q
                else:
                    # Fallback to Q-learning if next_action not available
                    target = exp.reward + self.gamma * self.q_table.v(exp.next_state)
            
            self.q_table.update_single(exp.state, exp.action, target)
            predictions.append(pred)
            targets.append(target)
        
        return {"critic_loss": self.loss(predictions, targets)}


def create_sarsa_agent(
    env: Any,
    *,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    bins: Union[int, Sequence[int]] = 10,
    clip_low: Sequence[float] | float = -4.8,
    clip_high: Sequence[float] | float = 4.8,
) -> QLearningAgent:
    """
    Factory to create SARSA (on-policy) agent by modifying Q-learning setup.
    
    SARSA vs Q-Learning differences:
    1. Uses on_policy=True flag (SARSA uses actual next actions)
       - Q-learning uses separate policies: behavior explores, target is greedy
       - SARSA uses same policy for both: learns about the policy it follows
    
    2. Uses on-policy critic updater
       - Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
       - SARSA: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    This makes SARSA learn about the policy it's actually following,
    including the effects of exploration, rather than assuming optimal play.
    """
    
    # Create the base agent with on_policy=True for SARSA behavior
    agent = create_qlearning_agent(
        env, 
        on_policy=True, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon, 
        bins=bins, 
        clip_low=clip_low, 
        clip_high=clip_high
    )
    
    return agent


if __name__ == "__main__":
    # Test SARSA implementation - same test as Q-learning for comparison
    import gym

    env = gym.make("CartPole-v1")
    agent = create_sarsa_agent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

    rewards: List[float] = []
    evaluations = []
    for episode in range(3000):
        # Use episode-based updates (Monte Carlo style) - best for SARSA
        total_reward = agent.learn(max_steps=env._max_episode_steps, update_frequency=-1)
        rewards.append(total_reward)

        # Evaluate every 250 episodes
        freq = 250
        if (episode + 1) % freq == 0:
            block = rewards[-freq:]
            avg = sum(block) / len(block)
            start = episode - freq + 1
            end = episode + 1
            evaluations.append(evaluate_policy(agent.policy, env, episodes=100))  # Fixed: use agent.policy

            print(f"SARSA - Average reward for episodes {start}–{end}: {avg:.2f}")
   
    # Print all the evaluations at the end
    print("SARSA Final evaluations:", evaluations)
    evaluation_score = evaluate_policy(agent.policy, env, episodes=100)  # Fixed: use agent.policy
    print("SARSA Evaluation score:", evaluation_score)
    env.close()