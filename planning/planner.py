"""
Planning module for GameMind using MCTS and LLM subgoal generation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .mcts import MCTSNode, MCTS


class Planner:
    """Planner that combines MCTS with LLM subgoal generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the planner."""
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # MCTS configuration
        self.mcts_simulations = config.get('mcts_simulations', 100)
        self.max_depth = config.get('max_depth', 10)
        self.exploration_constant = config.get('exploration_constant', 1.0)
        
        # Subgoal generation
        self.subgoal_enabled = config.get('subgoal_generation', {}).get('enabled', True)
        self.max_subgoals = config.get('subgoal_generation', {}).get('max_subgoals', 5)
        
        # Initialize LLM if available
        self.llm = None
        if self.subgoal_enabled:
            try:
                from llm.subgoal_generator import SubGoalGenerator
                self.llm = SubGoalGenerator(config)
            except ImportError:
                print("LLM not available, using basic planning only")
    
    def plan(self, observation, goal: str) -> List[str]:
        """Generate a plan for the given goal."""
        if not self.enabled:
            return []
        
        # Generate subgoals using LLM
        subgoals = self._generate_subgoals(observation, goal)
        
        # Use MCTS to refine the plan
        if subgoals:
            refined_plan = self._refine_plan_with_mcts(observation, subgoals)
            return refined_plan
        
        return []
    
    def _generate_subgoals(self, observation, goal: str) -> List[str]:
        """Generate subgoals using LLM."""
        if not self.llm or not self.subgoal_enabled:
            return self._generate_basic_subgoals(goal)
        
        try:
            subgoals = self.llm.generate_subgoals(goal, observation)
            return subgoals[:self.max_subgoals]
        except Exception as e:
            print(f"LLM subgoal generation failed: {e}")
            return self._generate_basic_subgoals(goal)
    
    def _generate_basic_subgoals(self, goal: str) -> List[str]:
        """Generate basic subgoals without LLM."""
        basic_subgoals = {
            'survive': ['find food', 'find shelter', 'avoid enemies'],
            'collect wood': ['find trees', 'chop wood', 'gather resources'],
            'make wood_pickaxe': ['collect wood', 'craft pickaxe', 'use workbench'],
            'place furnace': ['collect stone', 'find location', 'place building'],
            'defeat zombie': ['find weapon', 'approach enemy', 'attack']
        }
        
        for key, subgoals in basic_subgoals.items():
            if key in goal.lower():
                return subgoals
        
        return ['explore', 'gather resources', 'survive']
    
    def _refine_plan_with_mcts(self, observation, subgoals: List[str]) -> List[str]:
        """Refine the plan using MCTS."""
        if not subgoals:
            return []
        
        # Create MCTS tree
        root = MCTSNode(state=observation, action=None, parent=None)
        mcts = MCTS(exploration_constant=self.exploration_constant)
        
        # Run MCTS simulations
        for _ in range(self.mcts_simulations):
            node = mcts.select(root)
            if node.depth < self.max_depth:
                node = mcts.expand(node, subgoals)
            value = mcts.simulate(node)
            mcts.backpropagate(node, value)
        
        # Extract best plan
        best_plan = []
        current = root
        while current.children:
            best_child = max(current.children, key=lambda c: c.visits)
            if best_child.action:
                best_plan.append(best_child.action)
            current = best_child
        
        return best_plan if best_plan else subgoals
    
    def update_plan(self, current_state, executed_action, reward) -> Optional[List[str]]:
        """Update the plan based on execution results."""
        # Simple plan update logic
        if reward < 0:
            # Negative reward, consider replanning
            return self.plan(current_state, "recover from failure")
        return None 