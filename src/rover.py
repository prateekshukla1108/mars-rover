import os
import numpy as np
from collections import deque
import random
import warnings
import tensorflow as tf

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def setup_device():
    """Setup training device (GPU/CPU) based on availability"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return "GPU available. Using GPU for training."
        except RuntimeError as e:
            print(f"GPU error: {e}")
            print("Falling back to CPU")
            # Disable GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return "GPU error encountered. Using CPU for training."
    else:
        # No GPU found, use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "No GPU found. Using CPU for training."

# Setup device and get status message
DEVICE_STATUS = setup_device()

class MarsEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.reset()
    
    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        # Place obstacles (rocks) randomly
        num_obstacles = self.size // 2
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.grid[x, y] = 1
        
        # Place samples randomly
        num_samples = self.size // 3
        for _ in range(num_samples):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if self.grid[x, y] == 0:  # Only place on empty spots
                self.grid[x, y] = 2
        
        # Place rover at random empty position
        while True:
            self.rover_pos = (random.randint(0, self.size-1), 
                            random.randint(0, self.size-1))
            if self.grid[self.rover_pos] == 0:
                break
        
        self.samples_collected = 0
        return self._get_state()
    
    def _get_state(self):
        # Create state representation: rover's view of surroundings
        view_size = 3
        state = np.zeros((view_size, view_size))
        for i in range(view_size):
            for j in range(view_size):
                x = self.rover_pos[0] - 1 + i
                y = self.rover_pos[1] - 1 + j
                if 0 <= x < self.size and 0 <= y < self.size:
                    state[i, j] = self.grid[x, y]
        return state.flatten()
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = (self.rover_pos[0] + moves[action][0],
                  self.rover_pos[1] + moves[action][1])
        
        # Check boundaries
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return self._get_state(), -5, True  # Penalty for hitting boundary
        
        # Check collision with obstacle
        if self.grid[new_pos] == 1:
            return self._get_state(), -10, True  # Penalty for hitting obstacle
        
        # Move rover
        self.rover_pos = new_pos
        
        # Check if sample collected
        reward = -1  # Small penalty for each move
        done = False
        
        if self.grid[self.rover_pos] == 2:  # Sample
            reward = 20  # Reward for collecting sample
            self.grid[self.rover_pos] = 0  # Remove collected sample
            self.samples_collected += 1
            if self.samples_collected == self.size // 3:  # All samples collected
                done = True
                reward += 50  # Bonus for collecting all samples
        
        return self._get_state(), reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Enable mixed precision training if GPU is available
        if tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        
        # Use different optimizer settings based on device
        if tf.config.list_physical_devices('GPU'):
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                epsilon=1e-4,  # Smaller epsilon for mixed precision
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse',
                     optimizer=optimizer)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(env=None, episodes=100, max_steps=500, results_handler=None):
    # Device information
    device_info = {
        "status": DEVICE_STATUS,
        "gpu_devices": str(tf.config.list_physical_devices('GPU')),
        "memory_growth": str(tf.config.get_visible_devices())
    }
    
    print("\nDevice Information:")
    print(device_info["status"])
    if tf.config.list_physical_devices('GPU'):
        print(f"Using GPU: {tf.config.list_physical_devices('GPU')[0].name}")
    print("\nStarting Training...\n")
    if env is None:
        env = MarsEnvironment()
    
    # Training configuration
    config = {
        "episodes": episodes,
        "max_steps": max_steps,
        "state_size": 9,
        "action_size": 4,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epsilon_decay": 0.995,
        "gamma": 0.95
    }
    
    if results_handler:
        results_handler.log_config(config)
        results_handler.log_device_info(device_info)
    
    agent = DQNAgent(config["state_size"], config["action_size"])
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for time in range(max_steps):  # max steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            steps += 1
            
            if len(agent.memory) > config["batch_size"]:
                agent.replay(config["batch_size"])
                
            if done:
                if results_handler:
                    results_handler.log_episode(
                        episode_num=e+1,
                        total_reward=total_reward,
                        steps=steps,
                        samples_collected=env.samples_collected,
                        epsilon=agent.epsilon
                    )
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
                break

if __name__ == "__main__":
    train()
