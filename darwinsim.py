#Deep Q-Learning Agent for DarwinSim


import os
import random
import threading
import numpy as np
import pygame
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from PIL import Image, ImageTk
import queue
import datetime

# Add DummyAgent class at the top (after imports):
class DummyAgent:
    def __init__(self, rect, color, last_action=0, score=0):
        self.rect = rect
        self.color = color
        self.last_action = last_action
        self.score = score

# Simplify TensorFlow/Keras import logic:
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    print("TensorFlow or Keras could not be imported. Please install them to run this program. Try: pip install tensorflow keras")
    raise

# Use namedtuple for shared agent state
from collections import namedtuple
SharedAgentState = namedtuple('SharedAgentState', ['rect', 'score', 'last_action'])

# Replace shared_agent_states with a list of SharedAgentState
shared_agent_states = []
shared_agent_states_lock = threading.Lock()

# === Config Defaults ===
WIDTH, HEIGHT, GRID_SIZE, FPS = 600, 600, 20, 60
NUM_FOOD = 7
NUM_OBSTACLES = 5
STEPS_PER_EPISODE = 300
TARGET_UPDATE_FREQ = 5
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
MAX_AGENTS = 5

# Globals
EPISODES = 500
LEARNING_RATE = 0.0005  # Reduced from 0.001 for more stable learning
GAMMA = 0.99  # Increased from 0.97 for better long-term planning
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01  # Reduced from 0.05 for more exploration
EPSILON_DECAY_EPISODES = 300  # Increased from 200 for slower decay
PRIORITY_ALPHA = 0.6
PRIORITY_EPS = 1e-6
TARGET_UPDATE_FREQ = 10  # Increased from 5 for more stable target network
CLIP_NORM = 1.0  # Gradient clipping to prevent exploding updates

# Runtime flags
stop_flag = threading.Event()
use_pretrained = False
model_path = None
training_active = False
agent_colors = [(60, 0, 220), (60, 220, 60), (60, 60, 220),
                (220, 220, 60), (220, 220, 220)]  # Red, Green, Blue, Yellow, Purple
skip_ahead = False
# Flag that training threads read each step to decide if they should render
training_visualization_enabled = True

# Experience tuple
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# === Prioritized Replay Buffer ===
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, experience):
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        if self.size == 0:
            return []

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, np.array(weights))

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + PRIORITY_EPS
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# === Advanced Dueling DQN Model ===
def build_model(input_dim, output_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.1)(x)  # Add dropout for regularization
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    # Dueling architecture
    value = layers.Dense(128, activation='relu')(x)
    value = layers.Dense(1)(value)
    advantage = layers.Dense(128, activation='relu')(x)
    advantage = layers.Dense(output_dim)(advantage)
    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    model = tf.keras.models.Model(inputs=inputs, outputs=q_values)
    # Apply gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
    model.compile(optimizer=optimizer, loss='huber')
    return model

# === Advanced Agent ===
class DQNAgent:
    """Deep Q-Network Agent for DarwinSim."""
    def __init__(self, input_dim, output_dim, agent_id):
        self.agent_id = agent_id
        self.color = agent_colors[agent_id % len(agent_colors)]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = INITIAL_EPSILON
        self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, PRIORITY_ALPHA)
        self.beta = 0.4

        global use_pretrained, model_path
        if use_pretrained and model_path:
            try:
                self.model = models.load_model(model_path)
                self.target = models.load_model(model_path)
                print(f"Agent {agent_id}: Loaded pretrained model")
            except Exception as e:
                print(f"Agent {agent_id}: Failed to load model: {str(e)}")
                print("Creating new model...")
                self.model = build_model(input_dim, output_dim)
                self.target = build_model(input_dim, output_dim)
        else:
            self.model = build_model(input_dim, output_dim)
            self.target = build_model(input_dim, output_dim)

        self.target.set_weights(self.model.get_weights())
        self.reset()

    def reset(self):
        # Position agents in different starting locations
        if self.agent_id == 0:
            x = WIDTH // 4
            y = HEIGHT // 2
        elif self.agent_id == 1:
            x = WIDTH * 3 // 4
            y = HEIGHT // 2
        elif self.agent_id == 2:
            x = WIDTH // 2
            y = HEIGHT // 4
        elif self.agent_id == 3:
            x = WIDTH // 2
            y = HEIGHT * 3 // 4
        else:
            x = WIDTH // 2
            y = HEIGHT // 2

        self.rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        self.score = 0
        self.steps = 0
        self.last_action = 0
        self.food_collected = 0
        self.obstacles_hit = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)

        q_values = self.model.predict(state[np.newaxis])[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(Experience(state, action, reward, next_state, done))

    def replay(self):
        if self.memory.size < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(BATCH_SIZE, self.beta)

        # Double DQN update
        next_q_values = self.model.predict(next_states)
        best_actions = np.argmax(next_q_values, axis=1)

        target_next_q_values = self.target.predict(next_states)
        target_q_values = target_next_q_values[np.arange(BATCH_SIZE), best_actions]

        targets = rewards + GAMMA * target_q_values * (1 - dones)

        q_values = self.model.predict(states)
        q_values[np.arange(BATCH_SIZE), actions] = targets

        # Update priorities
        errors = np.abs(q_values[np.arange(BATCH_SIZE), actions] - targets)
        self.memory.update_priorities(indices, errors)

        # Train model
        history = self.model.fit(states, q_values, sample_weight=weights,
                                 batch_size=BATCH_SIZE, epochs=1, verbose=0)  # type: ignore[arg-type]

        # Update beta - slower annealing for more stable learning
        self.beta = min(1.0, self.beta + 0.0005)  # Reduced from 0.001

        # Safely access history
        if history is not None and hasattr(history, 'history') and 'loss' in history.history:
            return history.history['loss'][0]
        else:
            return 0.0

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def decay_epsilon(self, ep):
        drop = (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_EPISODES
        self.epsilon = max(FINAL_EPSILON, self.epsilon - drop)

    def save_model(self, path):
        try:
            self.model.save(path)
            print(f"Agent {self.agent_id}: Model saved to {path}")
        except Exception as e:
            print(f"Agent {self.agent_id}: Failed to save model: {e}")


# === Advanced World ===
class World:
    def __init__(self, visualize=False):
        self.visualize = visualize
        if visualize:
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("DarwinSim - Multi-Agent Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.food = []
        self.obstacles = []

    # --- Dynamic visualization control ---
    def enable_visualization(self):
        """Create a pygame window if not already present and turn on drawing."""
        if self.visualize and self.screen is not None:
            return  # Already enabled

        self.visualize = True
        if self.screen is None:
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("DarwinSim - Multi-Agent Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def disable_visualization(self):
        """Close pygame window and stop drawing to speed up training."""
        if not self.visualize and self.screen is None:
            return  # Already disabled

        self.visualize = False
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock = None
            self.font = None

    def spawn_food(self):
        self.food = []
        for _ in range(NUM_FOOD):
            while True:
                x = random.randrange(0, WIDTH, GRID_SIZE)
                y = random.randrange(0, HEIGHT, GRID_SIZE)
                rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)

                # Check for overlap with obstacles
                if not any(rect.colliderect(obs) for obs in self.obstacles):
                    self.food.append(rect)
                    break

    def spawn_obstacles(self):
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            while True:
                x = random.randrange(0, WIDTH, GRID_SIZE)
                y = random.randrange(0, HEIGHT, GRID_SIZE)
                rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)

                # Avoid spawning in center where agents start
                if (abs(x - WIDTH // 2) > GRID_SIZE * 3 or
                        abs(y - HEIGHT // 2) > GRID_SIZE * 3):
                    self.obstacles.append(rect)
                    break

    def move_food(self):
        # Move some food randomly
        for i in range(len(self.food)):
            if random.random() < 0.02:  # 2% chance to move each frame
                self.food[i].x = random.randrange(0, WIDTH, GRID_SIZE)
                self.food[i].y = random.randrange(0, HEIGHT, GRID_SIZE)

    def draw(self, agents):
        if not self.visualize or not self.screen:
            return

        self.screen.fill((20, 20, 35))  # Dark blue background

        # Draw grid
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (30, 30, 50), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (30, 30, 50), (0, y), (WIDTH, y), 1)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (70, 70, 100), obs)
            pygame.draw.rect(self.screen, (50, 50, 80), obs, 2)

        # Draw food
        for f in self.food:
            pygame.draw.rect(self.screen, (0, 200, 100), f)
            pygame.draw.rect(self.screen, (0, 150, 70), f, 2)

        # Draw agents
        for agent in agents:
            pygame.draw.rect(self.screen, agent.color, agent.rect)
            pygame.draw.rect(self.screen, (max(0, agent.color[0] - 40),
                                           max(0, agent.color[1] - 40),
                                           max(0, agent.color[2] - 40)), agent.rect, 2)

            # Draw eyes to indicate direction
            eye_size = GRID_SIZE // 5
            eye_offset = GRID_SIZE // 3
            directions = {
                0: (0, 0),  # No move
                1: (0, -eye_offset),  # Up
                2: (0, eye_offset),  # Down
                3: (-eye_offset, 0),  # Left
                4: (eye_offset, 0)  # Right
            }

            if agent.last_action in directions:
                dx, dy = directions[agent.last_action]
                eye_pos = (agent.rect.centerx + dx, agent.rect.centery + dy)
                pygame.draw.circle(self.screen, (255, 255, 255), eye_pos, eye_size)

        # Draw info
        if agents and self.font:
            agent = agents[0]
            score_text = self.font.render(f"Agent 0: Score: {agent.score}", True, agent_colors[0])
            self.screen.blit(score_text, (10, 10))

            if len(agents) > 1:
                agent = agents[1]
                score_text = self.font.render(f"Agent 1: Score: {agent.score}", True, agent_colors[1])
                self.screen.blit(score_text, (10, 30))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)


# === Training / Evaluation Logic ===
def run_agent_training(agent_id, params, status_queue, result_queue, log_queue):
    # Initialize variables to prevent reference errors
    episode_reward = 0.0
    avg_loss = 0.0
    plot_path = ""
    # --- TensorBoard writer ---
    log_dir = os.path.join("logs", f"agent_{agent_id}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir)

    # Extract parameters
    EPISODES = params['episodes']
    LEARNING_RATE = params['learning_rate']
    GAMMA = params['gamma']
    INITIAL_EPSILON = params['initial_epsilon']
    FINAL_EPSILON = params['final_epsilon']
    EPSILON_DECAY_EPISODES = params['epsilon_decay_episodes']
    visualize_every_n = params.get('visualize_every_n', 10)

    # Use global real-time visualization flag
    global training_visualization_enabled

    # Initialize world and agent – will visualize only for agent 0 when flag is on
    world = World(visualize=(agent_id == 0 and training_visualization_enabled))
    world.spawn_obstacles()

    # State: [agent_x, agent_y,
    #         dist_to_5_foods_x, dist_to_5_foods_y,
    #         dist_to_3_obstacles_x, dist_to_3_obstacles_y]
    input_dim = 2 + 5 * 2 + 3 * 2
    agent = DQNAgent(input_dim, 5, agent_id)  # 5 actions: none, up, down, left, right

    # Register agent in shared list for visualization
    global shared_agent_states, shared_agent_states_lock
    if agent_id == 0:
        with shared_agent_states_lock:
            shared_agent_states.clear()
            shared_agent_states.extend([None] * params['num_agents'])
    else:
        with shared_agent_states_lock:
            if len(shared_agent_states) < agent_id + 1:
                shared_agent_states.extend([None] * (agent_id + 1 - len(shared_agent_states)))

    # Initialize metrics
    rewards, scores, losses = [], [], []
    best_score = -np.inf
    start_time = time.time()

    for ep in range(1, EPISODES + 1):
        if stop_flag.is_set():
            log_queue.put(f"Agent {agent_id}: Training stopped by user")
            break

        # Reset environment
        world.spawn_food()
        agent.reset()
        episode_reward = 0
        episode_loss = 0
        update_count = 0

        for step in range(STEPS_PER_EPISODE):
            # Sync world visualization with latest user choice
            if agent_id == 0:
                if training_visualization_enabled and not world.visualize:
                    world.enable_visualization()
                elif not training_visualization_enabled and world.visualize:
                    world.disable_visualization()

            if stop_flag.is_set():
                break

            # Process events only if visualization is enabled for agent 0
            if agent_id == 0 and world.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_flag.set()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            global skip_ahead
                            skip_ahead = not skip_ahead
                            world.visualize = not skip_ahead

            # Only draw if world.visualize is True
            if agent_id == 0 and world.visualize:
                # Draw all agents' positions
                agents_to_draw = []
                with shared_agent_states_lock:
                    for idx, state in enumerate(shared_agent_states):
                        if state is not None:
                            dummy_agent = DummyAgent(state.rect, agent_colors[idx % len(agent_colors)], last_action=state.last_action, score=state.score)
                            agents_to_draw.append(dummy_agent)
                world.draw(agents_to_draw)

            # Get state
            state = get_agent_state(agent, world)

            # Agent action
            action = agent.act(state)
            agent.last_action = action
            agent.steps += 1  # Track total steps for reporting

            # Calculate distance to nearest food before and after move
            if world.food:
                nearest_food = min(world.food, key=lambda f: np.linalg.norm(np.array(agent.rect.center) - np.array(f.center)))
                old_dist = np.linalg.norm(np.array(agent.rect.center) - np.array(nearest_food.center))
            else:
                # No food left, maybe set a default value or skip food reward
                old_dist = 0

            # Move agent
            move_agent(agent, action)

            if world.food:
                new_dist = np.linalg.norm(np.array(agent.rect.center) - np.array(nearest_food.center))
                # Improved food shaping reward
                food_shaped_reward = (old_dist - new_dist) * 5.0  # Increased multiplier for better guidance
            else:
                food_shaped_reward = 0
                
            # Distance to nearest obstacle
            nearest_obs = min(world.obstacles, key=lambda o: np.linalg.norm(np.array(agent.rect.center) - np.array(o.center)))
            old_obs_dist = np.linalg.norm(np.array(agent.rect.center) - np.array(nearest_obs.center))
            obs_shaped_reward = (old_obs_dist - np.linalg.norm(np.array(agent.rect.center) - np.array(nearest_obs.center))) * -3.0  # Stronger penalty for getting closer
            # Reduced step penalty
            step_penalty = -0.02  # Reduced from -0.05
            reward = food_shaped_reward + obs_shaped_reward + step_penalty
            done = False
            # Check for food collection
            for i, food in enumerate(world.food[:]):
                if agent.rect.colliderect(food):
                    world.food.pop(i)
                    # Progressive reward based on remaining food
                    remaining_food = len(world.food)
                    food_reward = 30.0 + (remaining_food * 5.0)  # More reward for collecting food when more is available
                    reward += food_reward
                    agent.score += 1
                    agent.food_collected += 1
                    # Check if all food is collected
                    if not world.food:
                        reward += 100.0  # Bonus for completing the episode
                        done = True
                    break

            # Check for obstacle collision
            for obs in world.obstacles:
                if agent.rect.colliderect(obs):
                    reward -= 20.0  # Larger penalty for dying
                    agent.obstacles_hit += 1
                    done = True
                    break

            # Check boundary collision
            if (agent.rect.left < 0 or agent.rect.right > WIDTH or
                    agent.rect.top < 0 or agent.rect.bottom > HEIGHT):
                reward -= 20.0
                done = True

            # Get next state
            next_state = get_agent_state(agent, world)

            # Remember experience
            agent.remember(state, action, reward, next_state, done)

            # Experience replay
            loss = agent.replay()
            if loss > 0:
                episode_loss += loss
                update_count += 1

            # Move some food randomly
            if step % 10 == 0:
                world.move_food()

            # Update shared state for visualization
            with shared_agent_states_lock:
                shared_agent_states[agent_id] = SharedAgentState(agent.rect.copy(), agent.score, agent.last_action)

            # Update visualization
            if agent_id == 0 and world.visualize:
                # Draw all agents' positions
                agents_to_draw = []
                with shared_agent_states_lock:
                    for idx, state in enumerate(shared_agent_states):
                        if state is not None:
                            dummy_agent = DummyAgent(state.rect, agent_colors[idx % len(agent_colors)], last_action=state.last_action, score=state.score)
                            agents_to_draw.append(dummy_agent)
                world.draw(agents_to_draw)

            # Update metrics
            episode_reward += reward

            if done:
                # Add completion bonus for successful episodes
                if agent.score == NUM_FOOD:  # All food collected
                    reward += 50.0  # Additional completion bonus
                break

        # End of episode updates
        if update_count > 0:
            avg_loss = episode_loss / update_count
            losses.append(avg_loss)
        else:
            avg_loss = 0
            losses.append(0)

        # Update target network
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        # Decay epsilon
        agent.decay_epsilon(ep)

        # Save model if best score
        if agent.score > best_score:
            best_score = agent.score
            agent.save_model(f"best_model_agent{agent_id}.h5")

        # Update metrics
        rewards.append(episode_reward)
        scores.append(agent.score)

        # Calculate statistics
        avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        elapsed = time.time() - start_time
        time_per_episode = elapsed / ep

        # --- Plotting (every `visualize_every_n` episodes) ---
        if (ep % visualize_every_n == 0) or (ep == 1) or (ep == EPISODES):
            plt.figure(figsize=(6, 6))
            plt.subplot(3, 1, 1)
            plt.plot(rewards, 'b-')
            plt.title(f'Agent {agent_id} - Episode Rewards')
            plt.grid(True, alpha=0.3)

            plt.subplot(3, 1, 2)
            plt.plot(scores, 'g-')
            plt.title('Episode Scores')
            plt.grid(True, alpha=0.3)

            plt.subplot(3, 1, 3)
            plt.plot(losses, 'r-')
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.grid(True, alpha=0.3)

            # Save plot to image
            plot_path = f'agent_{agent_id}_training_plot.png'
            plt.savefig(plot_path, dpi=80)
            plt.close()

        # Send results to main thread
        result_data = {
            'agent_id': agent_id,
            'episode': ep,
            'score': agent.score,
            'reward': episode_reward,
            'loss': avg_loss,
            'epsilon': agent.epsilon,
            'plot_path': plot_path,
            'progress': ep / EPISODES * 100,
            'food_collected': agent.food_collected,
            'obstacles_hit': agent.obstacles_hit,
            'steps': agent.steps
        }
        result_queue.put(result_data)

        # Send status update
        status = (f"Agent {agent_id} | Episode {ep}/{EPISODES} | "
                  f"Score: {agent.score} | Reward: {episode_reward:.1f} | "
                  f"Avg Score: {avg_score:.1f} | Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        status_queue.put(status)

        # --- TensorBoard logging ---
        with writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward, step=ep)
            tf.summary.scalar('Episode Score', agent.score, step=ep)
            tf.summary.scalar('Training Loss', avg_loss, step=ep)
            tf.summary.scalar('Epsilon', agent.epsilon, step=ep)
        writer.flush()

    # Final update
    result_data = {
        'agent_id': agent_id,
        'episode': EPISODES,
        'score': agent.score,
        'reward': episode_reward,
        'loss': avg_loss,
        'epsilon': agent.epsilon,
        'plot_path': plot_path,
        'progress': 100,
        'food_collected': agent.food_collected,
        'obstacles_hit': agent.obstacles_hit,
        'steps': agent.steps
    }
    result_queue.put(result_data)

    # Save final model
    agent.save_model(f"final_model_agent{agent_id}.h5")

    # Close writer
    writer.close()
    log_queue.put(f"Agent {agent_id}: Training completed")

    return scores, rewards, losses


def get_agent_state(agent, world):
    # Normalized agent position
    state = [
        agent.rect.x / WIDTH,
        agent.rect.y / HEIGHT
    ]

    # Distances to closest 5 foods
    food_centers = np.array([np.array([food.x, food.y]) for food in world.food])
    agent_center = np.array([agent.rect.x, agent.rect.y])
    if len(food_centers) > 0:
        dists = np.linalg.norm(food_centers - agent_center, axis=1)
        sorted_idx = np.argsort(dists)
        for i in range(5):
            if i < len(sorted_idx):
                dx = (world.food[sorted_idx[i]].x - agent.rect.x) / WIDTH
                dy = (world.food[sorted_idx[i]].y - agent.rect.y) / HEIGHT
                state.extend([dx, dy])
            else:
                state.extend([0, 0])
    else:
        state.extend([0, 0] * 5)

    # Distances to closest 3 obstacles
    obs_centers = np.array([np.array([obs.x, obs.y]) for obs in world.obstacles])
    if len(obs_centers) > 0:
        dists = np.linalg.norm(obs_centers - agent_center, axis=1)
        sorted_idx = np.argsort(dists)
        for i in range(3):
            if i < len(sorted_idx):
                dx = (world.obstacles[sorted_idx[i]].x - agent.rect.x) / WIDTH
                dy = (world.obstacles[sorted_idx[i]].y - agent.rect.y) / HEIGHT
                state.extend([dx, dy])
            else:
                state.extend([0, 0])
    else:
        state.extend([0, 0] * 3)

    return np.array(state, dtype=np.float32)


def move_agent(agent, action):
    # Actions: 0=none, 1=up, 2=down, 3=left, 4=right
    if action == 1:  # Up
        agent.rect.y -= GRID_SIZE
    elif action == 2:  # Down
        agent.rect.y += GRID_SIZE
    elif action == 3:  # Left
        agent.rect.x -= GRID_SIZE
    elif action == 4:  # Right
        agent.rect.x += GRID_SIZE

    # Boundary check
    agent.rect.x = max(0, min(agent.rect.x, WIDTH - GRID_SIZE))
    agent.rect.y = max(0, min(agent.rect.y, HEIGHT - GRID_SIZE))


# === Advanced GUI with Multi-Agent Support ===
class DarwinSimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DarwinSim - Multi-Agent Training")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Initialize variables
        self.training_threads = []
        self.status_text = ""
        self.agent_results = {}
        self.active_agents = 1
        self.agent_plots = {}
        self.log_text = None

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        # Create main panels
        self.create_main_panels()

    def configure_styles(self):
        # Configure colors
        bg_color = "#2e2e2e"
        fg_color = "#e0e0e0"
        accent_color = "#4a9cff"

        self.style.configure('.', background=bg_color, foreground=fg_color, font=('Arial', 10))
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TButton', background="#3a3a3a", foreground=fg_color,
                             borderwidth=1, focusthickness=3, focuscolor='none')
        self.style.map('TButton', background=[('active', '#4a4a4a')])
        self.style.configure('TNotebook', background=bg_color)
        self.style.configure('TNotebook.Tab', background="#3a3a3a", foreground=fg_color,
                             padding=[10, 5], font=('Arial', 10, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', accent_color)])
        self.style.configure('TEntry', fieldbackground="#3a3a3a", foreground=fg_color)
        self.style.configure('TCombobox', fieldbackground="#3a3a3a", foreground=fg_color)
        self.style.configure('Vertical.TScrollbar', background="#3a3a3a")
        self.style.configure('Horizontal.TScrollbar', background="#3a3a3a")
        self.style.configure('Treeview', background="#3a3a3a", foreground=fg_color, fieldbackground="#3a3a3a")
        self.style.configure('Treeview.Heading', background=accent_color, foreground="white")
        self.style.configure('TProgressbar', troughcolor=bg_color, background=accent_color)

    def create_main_panels(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.setup_tab, text='Setup')
        self.notebook.add(self.training_tab, text='Training')
        self.notebook.add(self.evaluation_tab, text='Evaluation')
        self.notebook.add(self.logs_tab, text='Logs')
        self.notebook.add(self.comparison_tab, text='Comparison')

        # Setup tab content
        self.create_setup_tab()

        # Training tab content
        self.create_training_tab()

        # Evaluation tab content
        self.create_evaluation_tab()

        # Logs tab content
        self.create_logs_tab()

        # Comparison tab content
        self.create_comparison_tab()

    def create_setup_tab(self):
        # Parameters frame
        params_frame = ttk.LabelFrame(self.setup_tab, text="Training Parameters")
        params_frame.pack(fill='x', padx=10, pady=10)

        # Create parameter entries
        self.param_vars = {}
        params = [
            ('episodes', 'Episodes:', EPISODES),
            ('learning_rate', 'Learning Rate:', LEARNING_RATE),
            ('gamma', 'Gamma:', GAMMA),
            ('initial_epsilon', 'Initial Epsilon:', INITIAL_EPSILON),
            ('final_epsilon', 'Final Epsilon:', FINAL_EPSILON),
            ('epsilon_decay_episodes', 'Epsilon Decay Episodes:', EPSILON_DECAY_EPISODES),
            ('steps_per_episode', 'Steps per Episode:', STEPS_PER_EPISODE),
            ('batch_size', 'Batch Size:', BATCH_SIZE),
            ('memory_capacity', 'Memory Capacity:', MEMORY_CAPACITY),
            ('target_update_freq', 'Target Update Freq:', TARGET_UPDATE_FREQ),
            ('num_agents', 'Number of Agents (1-5):', 1),
            ('visualize_every_n', 'Visualize Every Nth Episode:', 10)
        ]

        for i, (name, label, default) in enumerate(params):
            frame = ttk.Frame(params_frame)
            frame.grid(row=i // 3, column=i % 3, sticky='w', padx=10, pady=5)
            ttk.Label(frame, text=label).pack(side='left', padx=(0, 5))
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(side='left')
            self.param_vars[name] = var

        # Checkbox for training visualization toggle
        self.visualize_training_var = tk.BooleanVar(value=True)
        vis_check = ttk.Checkbutton(params_frame,
                                    text="Enable Training Visualization",
                                    variable=self.visualize_training_var,
                                    command=self.on_visualize_toggle)
        # Position the checkbox in a new grid row
        vis_check.grid(row=(len(params) // 3) + 1, column=0, sticky='w', padx=10, pady=5)

        # Environment settings
        env_frame = ttk.LabelFrame(self.setup_tab, text="Environment Settings")
        env_frame.pack(fill='x', padx=10, pady=10)

        env_params = [
            ('width', 'Width:', WIDTH),
            ('height', 'Height:', HEIGHT),
            ('grid_size', 'Grid Size:', GRID_SIZE),
            ('fps', 'FPS:', FPS),
            ('num_food', 'Number of Food:', NUM_FOOD),
            ('num_obstacles', 'Number of Obstacles:', NUM_OBSTACLES)
        ]

        for i, (name, label, default) in enumerate(env_params):
            frame = ttk.Frame(env_frame)
            frame.grid(row=i // 3, column=i % 3, sticky='w', padx=10, pady=5)
            ttk.Label(frame, text=label).pack(side='left', padx=(0, 5))
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=8)
            entry.pack(side='left')
            self.param_vars[name] = var

        # Model management
        model_frame = ttk.LabelFrame(self.setup_tab, text="Model Management")
        model_frame.pack(fill='x', padx=10, pady=10)

        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Load Pretrained Model",
                    command=self.load_model).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset to New Model",
                    command=self.reset_model).pack(side='left', padx=5)

        self.model_status = ttk.Label(model_frame, text="Status: Using new model")
        self.model_status.pack(pady=5)

    # --- Callback for visualization toggle ---
    def on_visualize_toggle(self):
        """Update global flag so training threads can switch visualization on/off in real time."""
        global training_visualization_enabled
        training_visualization_enabled = self.visualize_training_var.get()

    def create_training_tab(self):
        # Control frame
        control_frame = ttk.LabelFrame(self.training_tab, text="Training Control")
        control_frame.pack(fill='x', padx=10, pady=10)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Start Training",
                   command=self.start_training).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Stop Training",
                   command=self.stop_training).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Save All Models",
                   command=self.save_all_models).pack(side='left', padx=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress")
        progress_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create scrolled frame for agent progress
        canvas = tk.Canvas(progress_frame, bg='#2e2e2e')
        scrollbar = ttk.Scrollbar(progress_frame, orient="vertical", command=canvas.yview)
        self.progress_frame = ttk.Frame(canvas)

        self.progress_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.progress_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Status display
        status_frame = ttk.Frame(self.training_tab)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Ready to train")
        self.status_label.pack(anchor='w')

    def create_evaluation_tab(self):
        # Evaluation controls
        eval_frame = ttk.LabelFrame(self.evaluation_tab, text="Evaluation Controls")
        eval_frame.pack(fill='x', padx=10, pady=10)

        btn_frame = ttk.Frame(eval_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Evaluate Best Models",
                   command=self.evaluate_models).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Run Visualization",
                   command=self.run_visualization).pack(side='left', padx=5)

        # Results display
        results_frame = ttk.LabelFrame(self.evaluation_tab, text="Evaluation Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create treeview for results
        columns = ('Agent', 'Score', 'Food Collected', 'Obstacles Hit', 'Steps', 'Epsilon')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings')

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # Add scrollbar to treeview
        tree_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

    def create_logs_tab(self):
        # Log display
        log_frame = ttk.LabelFrame(self.logs_tab, text="Training Logs")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD,
                                                  bg='#2e2e2e', fg='#e0e0e0',
                                                  font=('Courier', 10))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Log controls
        log_control_frame = ttk.Frame(self.logs_tab)
        log_control_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(log_control_frame, text="Clear Logs",
                   command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(log_control_frame, text="Save Logs",
                   command=self.save_logs).pack(side='left', padx=5)

    def create_comparison_tab(self):
        # Comparison controls
        comp_frame = ttk.LabelFrame(self.comparison_tab, text="Agent Comparison")
        comp_frame.pack(fill='x', padx=10, pady=10)

        btn_frame = ttk.Frame(comp_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(btn_frame, text="Compare All Agents",
                   command=self.compare_agents).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export Results",
                   command=self.export_results).pack(side='left', padx=5)

        # Comparison display
        self.comparison_frame = ttk.Frame(self.comparison_tab)
        self.comparison_frame.pack(fill='both', expand=True, padx=10, pady=10)

    def load_model(self):
        global use_pretrained, model_path
        filepath = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        if filepath:
            model_path = filepath
            use_pretrained = True
            self.model_status.config(text=f"Status: Using pretrained model - {os.path.basename(filepath)}")
            self.log_message(f"Loaded pretrained model: {filepath}")

    def reset_model(self):
        global use_pretrained, model_path
        use_pretrained = False
        model_path = None
        self.model_status.config(text="Status: Using new model")
        self.log_message("Reset to new model")

    def start_training(self):
        global stop_flag, training_active

        if training_active:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return

        try:
            # Get parameters
            params = {}
            for name, var in self.param_vars.items():
                if name in ['episodes', 'epsilon_decay_episodes', 'steps_per_episode',
                           'batch_size', 'memory_capacity', 'target_update_freq',
                           'width', 'height', 'grid_size', 'fps', 'num_food',
                           'num_obstacles', 'num_agents', 'visualize_every_n']:
                    params[name] = int(var.get())
                else:
                    params[name] = float(var.get())

            # Update global variables
            global WIDTH, HEIGHT, GRID_SIZE, FPS, NUM_FOOD, NUM_OBSTACLES
            global EPISODES, LEARNING_RATE, GAMMA, INITIAL_EPSILON, FINAL_EPSILON
            global EPSILON_DECAY_EPISODES, STEPS_PER_EPISODE, BATCH_SIZE, MEMORY_CAPACITY
            global visualize_every_n, TARGET_UPDATE_FREQ

            WIDTH = params['width']
            HEIGHT = params['height']
            GRID_SIZE = params['grid_size']
            FPS = params['fps']
            NUM_FOOD = params['num_food']
            NUM_OBSTACLES = params['num_obstacles']
            EPISODES = params['episodes']
            LEARNING_RATE = params['learning_rate']
            GAMMA = params['gamma']
            INITIAL_EPSILON = params['initial_epsilon']
            FINAL_EPSILON = params['final_epsilon']
            EPSILON_DECAY_EPISODES = params['epsilon_decay_episodes']
            STEPS_PER_EPISODE = params['steps_per_episode']
            BATCH_SIZE = params['batch_size']
            MEMORY_CAPACITY = params['memory_capacity']
            TARGET_UPDATE_FREQ = params['target_update_freq']
            visualize_every_n = params['visualize_every_n']

            self.active_agents = min(params['num_agents'], MAX_AGENTS)

            # Update global visualization flag based on current checkbox state
            global training_visualization_enabled
            training_visualization_enabled = self.visualize_training_var.get()

            # Clear previous results
            self.agent_results = {}
            self.clear_progress_displays()

            # Reset stop flag
            stop_flag.clear()
            training_active = True

            # Create queues for communication
            self.status_queue = queue.Queue()
            self.result_queue = queue.Queue()
            self.log_queue = queue.Queue()

            # Start training threads
            self.training_threads = []
            for agent_id in range(self.active_agents):
                thread = threading.Thread(
                    target=run_agent_training,
                    args=(agent_id, params, self.status_queue, self.result_queue, self.log_queue)
                )
                thread.daemon = True
                thread.start()
                self.training_threads.append(thread)

            # Start monitoring
            self.monitor_training()

            self.log_message(f"Started training {self.active_agents} agents")
            self.status_label.config(text=f"Training {self.active_agents} agents...")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
            training_active = False

    def stop_training(self):
        global stop_flag, training_active
        stop_flag.set()
        training_active = False
        self.status_label.config(text="Stopping training...")
        self.log_message("Training stopped by user")

    def clear_progress_displays(self):
        # Clear progress frame
        for widget in self.progress_frame.winfo_children():
            widget.destroy()

        # Clear agent plots
        self.agent_plots = {}

    def monitor_training(self):
        # Process status updates
        try:
            while not self.status_queue.empty():
                status = self.status_queue.get_nowait()
                self.status_label.config(text=status)
        except queue.Empty:
            pass

        # Process results
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_agent_display(result)
        except queue.Empty:
            pass

        # Process logs
        try:
            while not self.log_queue.empty():
                log_msg = self.log_queue.get_nowait()
                self.log_message(log_msg)
        except queue.Empty:
            pass

        # Check if training is still active
        if training_active:
            self.root.after(100, self.monitor_training)

    def update_agent_display(self, result):
        agent_id = result['agent_id']

        # Create or update agent display
        if agent_id not in self.agent_plots:
            self.create_agent_display(agent_id)

        # Update progress bar
        if f'progress_{agent_id}' in self.agent_plots:
            self.agent_plots[f'progress_{agent_id}']['value'] = result['progress']

        # Update statistics
        if f'stats_{agent_id}' in self.agent_plots:
            stats_text = (f"Episode: {result['episode']} | Score: {result['score']} | "
                         f"Reward: {result['reward']:.2f} | Loss: {result['loss']:.4f} | "
                         f"Epsilon: {result['epsilon']:.3f}")
            self.agent_plots[f'stats_{agent_id}'].config(text=stats_text)

        # Update plot if available
        if 'plot_path' in result and os.path.exists(result['plot_path']):
            self.update_agent_plot(agent_id, result['plot_path'])

        # Store result
        self.agent_results[agent_id] = result

    def create_agent_display(self, agent_id):
        # Create frame for this agent
        agent_frame = ttk.LabelFrame(self.progress_frame, text=f"Agent {agent_id}")
        agent_frame.pack(fill='x', padx=5, pady=5)

        # Progress bar
        progress = ttk.Progressbar(agent_frame, length=400, mode='determinate')
        progress.pack(fill='x', padx=5, pady=2)
        self.agent_plots[f'progress_{agent_id}'] = progress

        # Statistics label
        stats_label = ttk.Label(agent_frame, text="Initializing...")
        stats_label.pack(fill='x', padx=5, pady=2)
        self.agent_plots[f'stats_{agent_id}'] = stats_label

        # Plot frame
        plot_frame = ttk.Frame(agent_frame)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.agent_plots[f'plot_frame_{agent_id}'] = plot_frame

    def update_agent_plot(self, agent_id, plot_path):
        plot_frame = self.agent_plots.get(f'plot_frame_{agent_id}')
        if not plot_frame:
            return

        try:
            # Clear previous plot
            for widget in plot_frame.winfo_children():
                widget.destroy()

            # Load and display new plot
            img = Image.open(plot_path)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            label = ttk.Label(plot_frame, image=photo)
            label.pack()
            # Keep a reference to the image to prevent garbage collection
            if not hasattr(self, '_plot_images'):
                self._plot_images = {}
            self._plot_images[f'{agent_id}'] = photo

        except Exception as e:
            self.log_message(f"Error updating plot for agent {agent_id}: {str(e)}")

    def save_all_models(self):
        if not self.agent_results:
            messagebox.showwarning("Warning", "No training results to save!")
            return

        folder = filedialog.askdirectory(title="Select folder to save models")
        if folder:
            try:
                for agent_id in self.agent_results:
                    # Copy best model if it exists
                    src = f"best_model_agent{agent_id}.h5"
                    if os.path.exists(src):
                        dst = os.path.join(folder, f"best_model_agent{agent_id}.h5")
                        import shutil
                        shutil.copy2(src, dst)
                        self.log_message(f"Saved model for agent {agent_id}")

                messagebox.showinfo("Success", f"Models saved to {folder}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save models: {str(e)}")

    def evaluate_models(self):
        if not self.agent_results:
            messagebox.showwarning("Warning", "No training results to evaluate!")
            return

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add results to tree
        for agent_id, result in self.agent_results.items():
            self.results_tree.insert('', 'end', values=(
                f"Agent {agent_id}",
                result.get('score', 0),
                result.get('food_collected', 0),
                result.get('obstacles_hit', 0),
                result.get('steps', 0),
                f"{result.get('epsilon', 0):.3f}"
            ))

    def run_visualization(self):
        if not os.path.exists("best_model_agent0.h5"):
            messagebox.showwarning("Warning", "No trained model found! Train agents first.")
            return

        self.log_message("Starting visualization...")
        # This would start a separate visualization thread
        # For now, just show a message
        messagebox.showinfo("Info", "Visualization would run here with the best model")

    def compare_agents(self):
        if not self.agent_results:
            messagebox.showwarning("Warning", "No training results to compare!")
            return

        # Clear comparison frame
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Agent Comparison', fontsize=16)

        # Extract data for comparison
        agents = list(self.agent_results.keys())
        scores = [self.agent_results[agent]['score'] for agent in agents]
        food_collected = [self.agent_results[agent].get('food_collected', 0) for agent in agents]
        obstacles_hit = [self.agent_results[agent].get('obstacles_hit', 0) for agent in agents]
        steps = [self.agent_results[agent].get('steps', 0) for agent in agents]

        # Plot comparisons
        axes[0, 0].bar([f'Agent {i}' for i in agents], scores, color=[agent_colors[i] for i in agents])
        axes[0, 0].set_title('Final Scores')
        axes[0, 0].set_ylabel('Score')

        axes[0, 1].bar([f'Agent {i}' for i in agents], food_collected, color=[agent_colors[i] for i in agents])
        axes[0, 1].set_title('Food Collected')
        axes[0, 1].set_ylabel('Count')

        axes[1, 0].bar([f'Agent {i}' for i in agents], obstacles_hit, color=[agent_colors[i] for i in agents])
        axes[1, 0].set_title('Obstacles Hit')
        axes[1, 0].set_ylabel('Count')

        axes[1, 1].bar([f'Agent {i}' for i in agents], steps, color=[agent_colors[i] for i in agents])
        axes[1, 1].set_title('Steps Taken')
        axes[1, 1].set_ylabel('Steps')

        plt.tight_layout()

        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def export_results(self):
        if not self.agent_results:
            messagebox.showwarning("Warning", "No results to export!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                import csv
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Agent', 'Score', 'Food Collected', 'Obstacles Hit', 'Steps', 'Epsilon'])

                    for agent_id, result in self.agent_results.items():
                        writer.writerow([
                            agent_id,
                            result.get('score', 0),
                            result.get('food_collected', 0),
                            result.get('obstacles_hit', 0),
                            result.get('steps', 0),
                            result.get('epsilon', 0)
                        ])

                messagebox.showinfo("Success", f"Results exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def log_message(self, message):
        if self.log_text:
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)

    def clear_logs(self):
        if self.log_text:
            self.log_text.delete(1.0, tk.END)

    def save_logs(self):
        if not self.log_text:
            return

        filepath = filedialog.asksaveasfilename(
            title="Save logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {str(e)}")


def main():
    root = tk.Tk()
    app = DarwinSimGUI(root)

    def on_closing():
        global stop_flag
        stop_flag.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
