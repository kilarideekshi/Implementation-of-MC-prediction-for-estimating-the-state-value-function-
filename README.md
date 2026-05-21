# Implementation-of-MC-prediction-for-estimating-the-state-value-function-
## Date:
## Aim

To implement the Monte Carlo (MC) Prediction algorithm for estimating the state-value function of an agent interacting with an environment and to calculate and plot the estimated state-value function.

---

## Objective

- To understand Monte Carlo prediction in Reinforcement Learning.
- To estimate the value of each state using sampled episodes.
- To visualize the learned state-value function using a plot.

---

## Theory

Monte Carlo Prediction is a model-free reinforcement learning method used to estimate the value function of a policy. It learns directly from complete episodes of interaction with the environment.

The state-value function is defined as:

\[
V(s) = E[G_t \mid S_t = s]
\]

Where:

- \(V(s)\) = Value of state \(s\)
- \(G_t\) = Return obtained from time step \(t\)

Monte Carlo methods calculate the average return obtained after visiting a state multiple times.

---

## Algorithm

### Monte Carlo Prediction Algorithm

1. Initialize:
   - State-value function \(V(s)\)
   - Returns list for each state

2. Generate an episode using the given policy.

3. For each state appearing in the episode:
   - Calculate the return \(G\)
   - Store the return for that state
   - Update the value function using average return

4. Repeat for many episodes.

5. Plot the estimated state-value function.

---

## Program

```# Monte Carlo Prediction for State Value Function
# FrozenLake Environment

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
gamma = 0.9
episodes = 5000

# State Value Function
V = defaultdict(float)

# Returns storage
returns = defaultdict(list)

# Random Policy
def policy(state):
    return env.action_space.sample()

# Generate Episode
def generate_episode():

    episode = []

    # Reset environment
    state, info = env.reset()

    done = False

    while not done:

        action = policy(state)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store transition
        episode.append((state, action, reward))

        state = next_state

    return episode

# Monte Carlo Prediction
for ep in range(episodes):

    episode = generate_episode()

    G = 0
    visited_states = set()

    # Traverse episode backward
    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = gamma * G + reward

        # First-Visit MC
        if state not in visited_states:

            returns[state].append(G)

            V[state] = np.mean(returns[state])

            visited_states.add(state)

# Print State Values
print("\nState Value Function:\n")

for s in range(env.observation_space.n):
    print(f"State {s}: {V[s]:.3f}")

# Convert values to 4x4 grid
value_grid = np.zeros((4, 4))

for state in range(16):

    row = state // 4
    col = state % 4

    value_grid[row, col] = V[state]

# Plot Value Function
plt.figure(figsize=(6,6))

plt.imshow(value_grid, cmap='coolwarm')

# Add values on grid
for i in range(4):
    for j in range(4):

        plt.text(
            j,
            i,
            round(value_grid[i, j], 2),
            ha='center',
            va='center',
            color='black',
            fontsize=12
        )

plt.title("State Value Function Estimate")
plt.colorbar()
plt.show()
```
## Output
<img width="222" height="297" alt="image" src="https://github.com/user-attachments/assets/547d3746-2ead-4df9-9229-0f531563466d" />



### Output Graph

The following heatmap is generated for the estimated state-value function:

- Darker colors represent lower state values.
- Terminal states have value 0.
- States farther from terminal states have larger negative values.
<img width="665" height="467" alt="image" src="https://github.com/user-attachments/assets/3a707229-65e0-4dc7-993b-2a2abcc075b6" />



---
## Result

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the state-value function of the environment. The value of each state was calculated using sampled episodes, and the estimated state-value function was plotted successfully using a heatmap.




       

