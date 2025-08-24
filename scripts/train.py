#!/usr/bin/env python3
"""
GameMind Training Script
Main training script with curriculum learning.
"""

import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.crafter_wrapper import CrafterEnvWrapper
from agents.ppo_agent import PPOAgent
from planning.planner import Planner
from utils.logger import Logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GameMind agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration")
        return
    
    # Initialize logger
    logger = Logger(config)
    logger.logger.info("Starting GameMind training")
    
    # Create environment
    import crafter
    env = CrafterEnvWrapper(crafter.Env(), config)
    
    # Create agent
    agent = PPOAgent(env, config.get('agent', {}))
    
    # Create planner
    planner = Planner(config.get('planning', {}))
    
    # Curriculum stages
    curriculum = [
        {'name': 'Basic Survival', 'target': 'survive 25 steps', 'episodes': 25},
        {'name': 'Exploration', 'target': 'survive 50 steps', 'episodes': 25},
        {'name': 'Resource Collection', 'target': 'survive 75 steps', 'episodes': 25},
        {'name': 'Tool Crafting', 'target': 'survive 100 steps', 'episodes': 25},
        {'name': 'Building', 'target': 'survive 150 steps', 'episodes': 25},
        {'name': 'Advanced Tasks', 'target': 'survive 200 steps', 'episodes': 25}
    ]
    
    current_stage = 0
    stage_episodes = 0
    total_episodes = 0
    
    # Training loop
    while total_episodes < args.episodes:
        stage = curriculum[current_stage]
        logger.logger.info(f"Stage {current_stage + 1}: {stage['name']} - {stage['target']}")
        
        # Train for current stage
        for episode in range(stage['episodes']):
            if total_episodes >= args.episodes:
                break
                
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Generate subgoal for current stage
            if planner:
                subgoals = planner.plan(obs, stage['target'])
                if subgoals:
                    agent.set_subgoal(subgoals[0])
            
            # Episode loop
            while not done and episode_steps < 300:
                action = agent.act(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if done or truncated:
                    break
            
            # Update agent statistics
            agent.update_episode_stats(episode_reward, episode_steps)
            
            # Log progress
            if episode % 5 == 0:
                logger.logger.info(f"Episode {episode + 1}/{stage['episodes']} - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            # Train agent periodically
            if episode % 5 == 0:
                agent.train(total_timesteps=1000)
            
            total_episodes += 1
            stage_episodes += 1
        
        # Check if ready for next stage
        recent_rewards = agent.total_rewards[-20:] if len(agent.total_rewards) >= 20 else agent.total_rewards
        avg_reward = np.mean(recent_rewards)
        
        logger.logger.info(f"Stage {current_stage + 1} complete - Avg reward: {avg_reward:.2f}")
        
        # Progress to next stage
        current_stage = min(current_stage + 1, len(curriculum) - 1)
        stage_episodes = 0
        
        # Save checkpoint
        if total_episodes % 25 == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            agent.save_model(os.path.join(args.checkpoint_dir, f'checkpoint_{total_episodes}'))
    
    # Save final model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    agent.save_model(os.path.join(args.checkpoint_dir, 'final_model'))
    
    # Final statistics
    stats = agent.get_learning_stats()
    logger.logger.info(f"Training complete! Final avg reward: {stats.get('recent_avg_reward', 0):.2f}")


if __name__ == '__main__':
    main() 