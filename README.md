## 📄 `darwinsim.py` — Core Simulation Module

This is the main environment and agent logic for the Deep Q-learning simulation. It handles:

* 🌍 The 2D grid-based world with moving food and obstacles
* 🧠 The Dueling DQN agent architecture with prioritized replay
* 🎮 The real-time PyGame visualization system
* 🧪 The training loop and behavior shaping rewards
* 📊 TensorBoard logging for performance monitoring

---

### 🧠 Agent Architecture

Each agent uses a **Dueling Deep Q-Network (DQN)** with:

* Prioritized Experience Replay (PER)
* ε-greedy exploration with decay
* Double DQN target updates
* Gradient clipping & dropout regularization

Agents learn to survive, avoid obstacles, and collect moving food using continuous reinforcement.

---

### 🌍 Environment Overview

* Grid world: 600x600 pixels, 20px per grid square
* Randomly spawning **food** and **obstacles**
* Food has a chance to *move randomly* each frame
* Agents track position, direction, score, and last action

Visualization is powered by PyGame and can be toggled on/off in real time to speed up training.

---

### 🎯 Reward System

* ✅ Positive reward for collecting food
* ⚠️ Negative shaping reward when moving toward obstacles
* 🔀 Shaped rewards based on distance changes (to food or danger)
* 📈 Additional reward factors can be tuned in `run_agent_training()`

---

### ♻️ Training Loop

* Each agent trains in a **separate thread**
* `run_agent_training()`:

  * Initializes a world and a DQN agent
  * Runs `STEPS_PER_EPISODE` steps per episode
  * Logs reward, loss, food collection, and step count to TensorBoard
  * Syncs shared agent states for visualization
  * Periodically updates the target network

---

### 🧠 Key Classes & Functions

| Component                 | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `DQNAgent`                | The core deep learning agent with act, replay, and memory methods       |
| `World`                   | PyGame environment managing food, obstacles, and rendering              |
| `PrioritizedReplayBuffer` | Memory system for storing and sampling experiences based on TD error    |
| `run_agent_training()`    | Main training loop, supports visualization, reward shaping, and logging |
| `move_agent()`            | Handles grid-based movement per action taken                            |

---

### 📈 Visualization & Debugging

* Toggle PyGame visualization live using `training_visualization_enabled`
* View metrics in real-time with:

  ```bash
  tensorboard --logdir=logs/
  ```
* Debug unexpected behavior (like reward exploits) using the live PyGame view

---

### 🔬 Experimental Features

* Multiple agents with shared rendering state
* Distance-based reward shaping
* Threaded training with synchronized visualization
* Designed to evolve and support multi-agent behavior

