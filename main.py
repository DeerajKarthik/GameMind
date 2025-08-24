#!/usr/bin/env python3
"""
GameMind - Advanced AI Agent for Crafter Environment
Main entry point for training and evaluation.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

from envs.crafter_wrapper import CrafterEnvWrapper
from agents.ppo_agent import PPOAgent
from planning.planner import Planner
from utils.logger import Logger
from utils.eval import evaluate_agent
from llm.subgoal_generator import SubGoalGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main function for training and evaluation."""
    parser = argparse.ArgumentParser(description='GameMind Training and Evaluation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                       help='Mode: train or evaluate')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save/load checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration")
        return
    
    # Initialize logger
    logger = Logger(config)
    logger.logger.info("GameMind initialized")
    
    if args.mode == 'train':
        # Training mode
        logger.logger.info("Starting training mode")
        
        # Create environment
        import crafter
        env = CrafterEnvWrapper(crafter.Env(), config)
        
        # Create agent
        agent = PPOAgent(env, config.get('agent', {}))
        
        # Create planner
        planner = Planner(config.get('planning', {}))
        
        # Training loop
        for episode in range(args.episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.act(obs)
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            # Log progress
            if episode % 10 == 0:
                logger.logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
            
            # Train agent periodically
            if episode % 5 == 0:
                agent.train(total_timesteps=1000)
        
        # Save final model
        agent.save_model(os.path.join(args.checkpoint_dir, 'final_model'))
        logger.logger.info("Training completed")
        
    else:
        # Evaluation mode
        logger.logger.info("Starting evaluation mode")
        
        # Create environment
        import crafter
        env = CrafterEnvWrapper(crafter.Env(), config)
        
        # Load trained agent
        agent = PPOAgent(env, config.get('agent', {}))
        agent.load_model(os.path.join(args.checkpoint_dir, 'final_model'))
        
        # Evaluate agent
        results = evaluate_agent(agent, env, num_episodes=10)
        logger.logger.info(f"Evaluation results: {results}")


if __name__ == '__main__':
    main() 