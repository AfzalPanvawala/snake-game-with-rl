# Snake Game with Reinforcement Learning (Double DQN)

An autonomous **Snake** agent trained with **Deep Reinforcement Learning (Double DQN)** in Python and PyTorch, featuring a continuous wrap-around grid, Manhattan distance reward shaping, real-time Pygame-CE visualization with directional eyes on the snake's head, and Matplotlib accuracy curves.

## Highlights
- **Continuous Non-Episodic Gameplay:** Runs on a wrap-around grid (`10x10`), growing permanently with every apple eaten until all grid cells are filled.
- **Double DQN & Experience Replay:** Implemented in PyTorch with target network updates to reduce Q-value overestimation and stabilize learning.
- **Reward Shaping & Accuracy Tracking:** Uses Manhattan distance feedback to guide the snake toward food and tracks steps-per-food efficiency in Matplotlib.
- **Live Pygame-CE Rendering:** Shows live grid-fill progress, food count, and directional head eyes while training.

## How to Run
1. Install the required Python packages:
   ```bash
   pip install torch numpy matplotlib pygame-ce
