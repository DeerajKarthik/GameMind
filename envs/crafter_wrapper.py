"""
Crafter environment wrapper for GameMind with reward shaping.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional


class CrafterEnvWrapper:
    """Gym-compatible wrapper for Crafter environment with reward shaping."""
    
    def __init__(self, env, config: Dict[str, Any]):
        """Initialize the wrapper."""
        self.env = env
        self.config = config
        
        # Environment state tracking
        self.prev_achievements = None
        self.prev_explored = 0
        self.prev_health = None
        self.prev_food = None
        self.prev_drink = None
        self.prev_inventory = {}
        self.prev_pos = None
        self.step_count = 0
        
        # Achievement tracking
        self.achievements = [
            'collect_wood', 'collect_stone', 'collect_coal', 'collect_iron',
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'place_furnace', 'place_table', 'defeat_zombie', 'defeat_skeleton'
        ]
        
        # Action space
        if hasattr(env, 'action_space'):
            n = env.action_space.n
            self.action_space = gym.spaces.Discrete(n)
        else:
            self.action_space = gym.spaces.Discrete(17)
        
        # Observation space
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    
    def step(self, action):
        """Execute action and return observation with shaped reward."""
        obs, reward, done, info = self.env.step(action)
        obs = self._transpose_obs(obs)
        
        if hasattr(self, 'config') and isinstance(self.config, dict) and self.config.get('env', {}).get('reward_shaping', False):
            reward = self._shape_reward(obs, reward, info)
        
        return obs, reward, done, False, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        crafter_kwargs = {k: v for k, v in kwargs.items() if k not in ['seed', 'options']}
        obs = self.env.reset(**crafter_kwargs)
        obs = self._transpose_obs(obs)
        self.prev_achievements = None
        self.prev_explored = 0
        self.prev_health = None
        self.prev_food = None
        self.prev_drink = None
        self.prev_inventory = {}
        self.prev_pos = None
        self.step_count = 0
        return obs, {}
    
    def _transpose_obs(self, obs):
        """Convert HWC to CHW format."""
        return np.transpose(obs, (2, 0, 1))
    
    def _shape_reward(self, obs, reward, info):
        """Shape the reward for better learning."""
        shaped = 0.0
        
        # Achievement rewards
        if self.prev_achievements is not None:
            for ach in self.achievements:
                if info.get(ach, 0) > self.prev_achievements.get(ach, 0):
                    shaped += 5.0
                    
                    task_rewards = self.config.get('task_rewards', {})
                    if ach in task_rewards:
                        shaped += task_rewards[ach]
                    else:
                        if ach in ['collect_wood', 'collect_stone', 'collect_coal', 'collect_iron']:
                            shaped += 3.0
                        elif ach in ['make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe']:
                            shaped += 8.0
                        elif ach in ['place_furnace', 'place_table']:
                            shaped += 6.0
                        elif ach in ['defeat_zombie', 'defeat_skeleton']:
                            shaped += 4.0
                    
        self.prev_achievements = {ach: info.get(ach, 0) for ach in self.achievements}
        
        # Exploration rewards
        explored = info.get('explore', 0)
        if self.prev_explored is not None and explored > self.prev_explored:
            exploration_gain = explored - self.prev_explored
            shaped += exploration_gain * 0.5
            if exploration_gain >= 10:
                shaped += 2.0
        self.prev_explored = explored
        
        # Survival rewards
        health = info.get('health', 0)
        if self.prev_health is not None:
            health_change = health - self.prev_health
            if health_change > 0:
                shaped += health_change * 0.2
            elif health_change < 0:
                shaped -= abs(health_change) * 0.1
        self.prev_health = health
        
        food = info.get('food', 0)
        if self.prev_food is not None:
            food_change = food - self.prev_food
            if food_change > 0:
                shaped += food_change * 0.2
            elif food_change < 0:
                shaped -= abs(food_change) * 0.1
        self.prev_food = food
        
        drink = info.get('drink', 0)
        if self.prev_drink is not None:
            drink_change = drink - self.prev_drink
            if drink_change > 0:
                shaped += drink_change * 0.2
            elif drink_change < 0:
                shaped -= abs(drink_change) * 0.1
        self.prev_drink = drink
        
        # Inventory progress bonus
        inventory = info.get('inventory', {})
        if hasattr(self, 'prev_inventory'):
            for item, count in inventory.items():
                if item in self.prev_inventory:
                    if count > self.prev_inventory[item]:
                        shaped += 0.5
                else:
                    shaped += 1.0
        self.prev_inventory = inventory.copy()
        
        # Position-based rewards
        player_pos = info.get('player_pos', None)
        if hasattr(self, 'prev_pos') and player_pos is not None:
            try:
                if isinstance(player_pos, (list, np.ndarray)) and isinstance(self.prev_pos, (list, np.ndarray)):
                    if not np.array_equal(player_pos, self.prev_pos):
                        shaped += 0.1
                elif isinstance(player_pos, (list, np.ndarray)) or isinstance(self.prev_pos, (list, np.ndarray)):
                    shaped += 0.1
                elif player_pos != self.prev_pos:
                    shaped += 0.1
            except (ValueError, TypeError):
                shaped += 0.1
        self.prev_pos = player_pos
        
        # Time-based survival bonus
        self.step_count += 1
        if self.step_count % 50 == 0:
            shaped += 1.0
        
        # Clip reward to prevent extreme values
        shaped = np.clip(shaped, -10.0, 20.0)
        
        return reward + shaped
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        self.env.close() 