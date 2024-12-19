import os
import json
import csv
import time
import matplotlib.pyplot as plt
from datetime import datetime

class ResultsHandler:
    def __init__(self, base_dir='results'):
        # Get the root directory (directory of this script)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set base directory within the root directory
        self.base_dir = os.path.join(root_dir, base_dir)
        
        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup directory structure
        self.run_dir = os.path.join(self.base_dir, f'run_{self.timestamp}')
        self.ensure_directories()
        
        # Initialize results storage
        self.episode_results = []
        self.training_config = {}
        self.device_info = {}
        
        # Setup files
        self.csv_path = os.path.join(self.run_dir, 'episode_results.csv')
        self.json_path = os.path.join(self.run_dir, 'training_info.json')
        self.plot_path = os.path.join(self.run_dir, 'training_plots.png')
        
        # Initialize CSV file
        self._init_csv()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.run_dir, exist_ok=True)
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward', 'Steps', 'Samples_Collected', 
                           'Epsilon', 'Time_Taken'])
    
    def log_config(self, config):
        """Log training configuration"""
        self.training_config = config
        self._save_json()
    
    def log_device_info(self, device_info):
        """Log device (GPU/CPU) information"""
        self.device_info = device_info
        self._save_json()
    
    def log_episode(self, episode_num, total_reward, steps, samples_collected, epsilon):
        """Log results from a single episode"""
        time_stamp = time.strftime('%H:%M:%S')
        
        # Store results
        episode_data = {
            'Episode': episode_num,
            'Total_Reward': total_reward,
            'Steps': steps,
            'Samples_Collected': samples_collected,
            'Epsilon': epsilon,
            'Time_Taken': time_stamp
        }
        
        self.episode_results.append(episode_data)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_num, total_reward, steps, samples_collected, 
                epsilon, time_stamp
            ])
    
    def _save_json(self):
        """Save configuration and device info to JSON"""
        data = {
            'timestamp': self.timestamp,
            'training_config': self.training_config,
            'device_info': self.device_info
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def generate_plots(self):
        """Generate and save training visualization plots"""
        if not self.episode_results:
            return
        
        episodes = [r['Episode'] for r in self.episode_results]
        rewards = [r['Total_Reward'] for r in self.episode_results]
        samples = [r['Samples_Collected'] for r in self.episode_results]
        epsilons = [r['Epsilon'] for r in self.episode_results]
        
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(3, 1, 1)
        plt.plot(episodes, rewards, 'b-')
        plt.title('Training Progress')
        plt.ylabel('Total Reward')
        
        # Plot samples collected
        plt.subplot(3, 1, 2)
        plt.plot(episodes, samples, 'g-')
        plt.ylabel('Samples Collected')
        
        # Plot epsilon decay
        plt.subplot(3, 1, 3)
        plt.plot(episodes, epsilons, 'r-')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
    
    def save_final_summary(self):
        """Generate and save final training summary"""
        if not self.episode_results:
            return
        
        summary = {
            'total_episodes': len(self.episode_results),
            'avg_reward': sum(r['Total_Reward'] for r in self.episode_results) / len(self.episode_results),
            'max_reward': max(r['Total_Reward'] for r in self.episode_results),
            'min_reward': min(r['Total_Reward'] for r in self.episode_results),
            'avg_samples': sum(r['Samples_Collected'] for r in self.episode_results) / len(self.episode_results),
            'max_samples': max(r['Samples_Collected'] for r in self.episode_results),
            'final_epsilon': self.episode_results[-1]['Epsilon']
        }
        
        summary_path = os.path.join(self.run_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Generate plots for the final state
        self.generate_plots()
