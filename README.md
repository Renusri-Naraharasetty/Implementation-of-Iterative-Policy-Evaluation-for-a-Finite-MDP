# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python


import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Parameters
gamma = 0.99
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions


# Initialize value function
V = np.zeros(n_states)

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """

    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0
        new_V = np.copy(V)

        for state in range(env.observation_space.n):

            value = 0

            for action in range(env.action_space.n):

                action_prob = policy[state][action]

                for prob, next_state, reward, done in env.P[state][action]:

                    value += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            new_V[state] = value
            delta = max(delta, abs(V[state] - new_V[state]))

        V = new_V
        iteration += 1

        if delta < theta:
            break

    return V, iteration

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: RENUSRI NARAHARASHETTY")
print("Register Number: 212223240139")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name: RENUSRI NARAHARASHETTY")
print("Register Number: 212223240139")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()


```

---

## Output

FOR GAMMA=0.99

<img width="748" height="366" alt="image" src="https://github.com/user-attachments/assets/8cd66f6a-f054-4663-9ea6-d0fe675b01f9" />

FOR GAMMA=0.9

<img width="767" height="372" alt="image" src="https://github.com/user-attachments/assets/1e35181f-b060-4a1b-82e5-e57cc321bffb" />


---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text

A higher gamma values future rewards more, causing values to propagate further and become larger;
The zeros, is because those specific states are holes (traps) or
starting positions from which the agent can never reach the goal under a random policy.
```




---


