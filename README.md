# DQN for Pong with Saliency Map

## Overview
This project implements a Deep Q-Learning (DQN) model to train an AI agent to play a custom Pong game. The innovative use of saliency maps generated through integrated gradient helps visualize and interpret the model’s decision-making process.

## Game Environment
The Pong game is developed using the pygame library, which provides necessary functionalities for game development like rendering graphics and handling user inputs. The state of the game includes the positions of the paddle and ball and their velocities, captured as an image and preprocessed to reduce dimensionality for faster computation.

## Reward System
- **+10 points**: When the ball goes beyond the left boundary (opponent misses).
- **-10 points**: When the ball goes beyond the right boundary (agent misses).
- **+5 points**: When the ball hits the right paddle (successful interception).

## State Representation
The images are preprocessed by converting them to grayscale and reducing their size to (84, 84). Multiple frames are stacked together to give the model a sense of motion and dynamics, helping it understand the direction and velocity of the ball.

## Deep Q-Network (DQN) Agent
The DQN agent uses a Convolutional Neural Network (CNN) to approximate the Q-value function, representing the expected future rewards of taking a particular action in a given state. The architecture includes:

- **Input Layer**: Accepts preprocessed stacked frames.
- **Convolutional Layers**: Extract features and capture complex patterns.
- **Flattening Layer**: Transforms the multidimensional output into a single vector.
- **Fully Connected Layer**: Interprets features and outputs Q-values for each action.
- **Output Layer**: Produces final Q-values for actions (e.g., moving the paddle up or down).

## Replay Buffer
A replay buffer stores transitions observed by the agent, breaking the correlation between consecutive samples and stabilizing the learning process.

## Training the Model
The agent uses an ε-greedy policy for exploration and exploitation, starting with a high exploration factor (ε) that gradually decays. The Q-values are updated using the Bellman equation. Key parameters include:

- **Gamma (γ = 0.99)**: Discount factor for future rewards.
- **Learning Rate (α = 0.0001)**: Determines the update rate for neural network weights.
- **Epsilon (ε = 1.0)**: Exploration factor, gradually decayed.
- **Decay Rate (25e-7)**: Slow decay of epsilon to ensure continued exploration.
- **Minimum Epsilon (0.1)**: Prevents epsilon from decaying to zero.

The target network, a copy of the Q-network, generates Q-value targets and is updated less frequently, stabilizing learning.

## Loss Function
The loss function is the mean squared error between predicted and target Q-values:

\[ \text{Loss} = \frac{1}{N} \sum (\text{targetQ} - Q(s, a; \theta))^2 \]

where \(\text{targetQ} = r + \gamma \max_{a'} Q(s', a'; \theta^-)\) if not done, otherwise \(\text{targetQ} = r\).

## Saliency Maps
Saliency maps are used to visually explain which parts of the input frame the network considers important when deciding on an action. This is achieved using Integrated Gradients which attributes the prediction of a deep network to its input features by integrating gradients along the path from a given baseline (typically a zero input) to the actual input, thus highlighting the input pixels' contributions to the neural network's decisions.

## Integrated Gradients Overview
- **Choosing a Baseline**: Typically a zero vector for images.
- **Path Interpolation**: Series of steps from baseline to actual input.
- **Computing Gradients**: Gradients of model output at each step.
- **Integrate Gradients**: Aggregates gradients across steps to compute feature attribution.

## Project Screenshots
![image](https://github.com/hari1016/DQN-with-saliency-map/assets/63118506/a52a0142-71a7-4a9a-81c8-33cfeee6ca0f)

