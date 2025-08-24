"""
Evaluation utilities for GameMind.
"""

import numpy as np
from typing import Dict, List, Any


def evaluate_agent(agent, env, num_episodes: int = 10) -> List[Dict[str, Any]]:
    """Evaluate an agent over multiple episodes."""
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 300:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if done or truncated:
                break
        
        # Collect episode results
        episode_result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'achievements': _extract_achievements(info),
            'info': info
        }
        
        results.append(episode_result)
    
    return results


def _extract_achievements(info: Dict[str, Any]) -> Dict[str, int]:
    """Extract achievement information from episode info."""
    achievement_keys = [
        'collect_wood', 'collect_stone', 'collect_coal', 'collect_iron',
        'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
        'place_furnace', 'place_table', 'defeat_zombie', 'defeat_skeleton'
    ]
    
    achievements = {}
    for key in achievement_keys:
        if key in info:
            achievements[key] = info[key]
    
    return achievements


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics from results."""
    if not results:
        return {}
    
    rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]
    
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'min_steps': np.min(steps),
        'max_steps': np.max(steps),
        'total_episodes': len(results)
    }
    
    return metrics


def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Print a summary of evaluation results."""
    if not results:
        print("No evaluation results to display.")
        return
    
    metrics = calculate_metrics(results)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Episodes: {metrics['total_episodes']}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print(f"Mean Steps: {metrics['mean_steps']:.2f} ± {metrics['std_steps']:.2f}")
    print(f"Steps Range: [{metrics['min_steps']:.2f}, {metrics['max_steps']:.2f}]")
    
    # Achievement summary
    all_achievements = {}
    for result in results:
        for ach, count in result['achievements'].items():
            if ach not in all_achievements:
                all_achievements[ach] = []
            all_achievements[ach].append(count)
    
    if all_achievements:
        print("\nAchievements:")
        for ach, counts in all_achievements.items():
            mean_count = np.mean(counts)
            if mean_count > 0:
                print(f"  {ach}: {mean_count:.2f} ± {np.std(counts):.2f}")
    
    print("="*50) 