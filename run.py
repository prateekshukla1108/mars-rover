#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from src.rover import train, MarsEnvironment
from src.results_handler import ResultsHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_args():
    parser = argparse.ArgumentParser(description='Mars Rover RL Training Script')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Size of the environment grid (default: 10)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--render', action='store_true',
                       help='Enable basic console rendering of environment')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    return parser.parse_args()

def render_environment(env):
    """Simple console-based rendering of the environment"""
    grid = env.grid.copy()
    grid[env.rover_pos] = 8  # Mark rover position with 8
    
    # Create a visualization mapping
    symbols = {
        0: '·',  # Empty space
        1: '▲',  # Obstacle
        2: '◆',  # Sample
        8: '█'   # Rover
    }
    
    # Print the grid
    print('\n' + '=' * (env.size * 2 + 1))
    for row in grid:
        print('|', end='')
        for cell in row:
            print(f'{symbols[cell]}|', end='')
        print()
    print('=' * (env.size * 2 + 1))
    print(f'Samples collected: {env.samples_collected}')
    print()

def main():
    # Get command line arguments
    args = setup_args()
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Add project root to Python path
    sys.path.append(project_root)
    
    # Initialize results handler
    results_handler = ResultsHandler(base_dir=args.results_dir)
    
    # Print starting messages
    print("Starting Mars Rover RL Training...")
    print(f"Episodes: {args.episodes}")
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Max Steps per Episode: {args.max_steps}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print("\nPress Ctrl+C to stop training\n")
    
    try:
        # Initialize environment with specified grid size
        env = MarsEnvironment(size=args.grid_size)
        
        # If rendering is enabled, monkey patch the step method
        if args.render:
            original_step = env.step
            def step_with_render(action):
                state, reward, done = original_step(action)
                render_environment(env)
                return state, reward, done
            env.step = step_with_render
            
            # Also render initial state
            render_environment(env)
        
        # Start training
        train(env=env, episodes=args.episodes, max_steps=args.max_steps,
              results_handler=results_handler)
        
        # Save final results
        results_handler.save_final_summary()
        print(f"\nTraining complete! Results saved in: {results_handler.run_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Still save results even if interrupted
        results_handler.save_final_summary()
        print(f"Partial results saved in: {results_handler.run_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)        
        # Start training
        train(env=env, episodes=args.episodes, max_steps=args.max_steps)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
