"""
PPO Agent implementation using Stable-Baselines3 for GameMind.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


class PPOAgent:
    """PPO Agent using Stable-Baselines3 with enhanced features."""
    
    def __init__(self, env, config: dict):
        """Initialize the PPO agent."""
        self.env = DummyVecEnv([lambda: env])
        self.config = config
        self.model = PPO('CnnPolicy', self.env, verbose=1, **self._get_sb3_args())
        
        # Training tracking
        self.episode_count = 0
        self.total_rewards = []
        self.learning_progress = []
        self.current_subgoal = None
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _get_sb3_args(self):
        """Get Stable-Baselines3 PPO arguments."""
        agent_config = self.config.get('agent', {})
        return {
            'learning_rate': agent_config.get('learning_rate', 0.0003),
            'gamma': agent_config.get('gamma', 0.99),
            'n_steps': agent_config.get('n_steps', 2048),
            'batch_size': agent_config.get('batch_size', 64),
            'n_epochs': agent_config.get('n_epochs', 10),
            'gae_lambda': agent_config.get('gae_lambda', 0.95),
            'clip_range': agent_config.get('clip_range', 0.2),
            'clip_range_vf': None,
            'ent_coef': agent_config.get('ent_coef', 0.01),
            'vf_coef': agent_config.get('vf_coef', 0.5),
            'max_grad_norm': agent_config.get('max_grad_norm', 0.5),
            'policy_kwargs': agent_config.get('policy_kwargs', {
                'net_arch': {
                    'pi': [256, 256],
                    'vf': [256, 256]
                },
                'activation_fn': torch.nn.ReLU,
                'ortho_init': True,
            })
        }
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        eval_env = DummyVecEnv([lambda: self.env.envs[0]])
        agent_config = self.config.get('agent', {})
        n_steps = agent_config.get('n_steps', 2048)
        
        self.eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",
            log_path="./logs/",
            eval_freq=max(n_steps // 4, 1),
            deterministic=True,
            render=False
        )
        
        self.checkpoint_callback = CheckpointCallback(
            save_freq=max(n_steps // 2, 1),
            save_path="./checkpoints/",
            name_prefix="ppo_agent"
        )
    
    def set_subgoal(self, subgoal):
        """Set current subgoal for hierarchical learning."""
        self.current_subgoal = subgoal
        if hasattr(self, 'current_subgoal'):
            self.learning_progress.append({
                'episode': self.episode_count,
                'subgoal_change': True,
                'new_subgoal': subgoal
            })
    
    def train(self, total_timesteps):
        """Train the agent with callbacks."""
        agent_config = self.config.get('agent', {})
        initial_lr = agent_config.get('learning_rate', 0.0003)
        
        callbacks = [self.eval_callback, self.checkpoint_callback]
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        self.learning_progress.append({
            'episode': self.episode_count,
            'training_completed': True,
            'total_timesteps': total_timesteps
        })
    
    def act(self, obs):
        """Get action from the agent with adaptive exploration."""
        if self.episode_count < 100:
            deterministic = False
        elif self.episode_count < 500:
            deterministic = np.random.random() < 0.7
        else:
            deterministic = True
        
        if self.current_subgoal:
            action, _ = self.model.predict(obs, deterministic=deterministic)
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # Validate action
        if isinstance(action, np.ndarray):
            action = action.item()
        
        return action
    
    def update_episode_stats(self, episode_reward, episode_steps):
        """Update episode statistics and learning progress."""
        self.episode_count += 1
        self.total_rewards.append(episode_reward)
        
        self.learning_progress.append({
            'episode': self.episode_count,
            'reward': episode_reward,
            'steps': episode_steps,
            'avg_reward': np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
        })
    
    def get_learning_stats(self):
        """Get current learning statistics."""
        if not self.total_rewards:
            return {}
        
        recent_rewards = self.total_rewards[-100:]
        return {
            'episode_count': self.episode_count,
            'total_rewards': self.total_rewards,
            'recent_avg_reward': np.mean(recent_rewards),
            'best_reward': max(self.total_rewards),
            'learning_progress': self.learning_progress
        }
    
    def save_model(self, path):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model."""
        if os.path.exists(path):
            self.model = PPO.load(path, env=self.env)
        else:
            print(f"Model not found at {path}") 