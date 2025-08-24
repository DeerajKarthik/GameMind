"""
Monte Carlo Tree Search implementation for GameMind.
"""

import numpy as np
import math
from typing import List, Optional, Any


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, action=None, parent=None):
        """Initialize MCTS node."""
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
    
    def add_child(self, state, action):
        """Add a child node."""
        child = MCTSNode(state, action, self)
        self.children.append(child)
        return child
    
    def is_fully_expanded(self, available_actions):
        """Check if node is fully expanded."""
        return len(self.children) >= len(available_actions)
    
    def get_ucb_value(self, exploration_constant):
        """Calculate UCB value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search algorithm."""
    
    def __init__(self, exploration_constant: float = 1.0):
        """Initialize MCTS."""
        self.exploration_constant = exploration_constant
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCB."""
        while node.children:
            if not node.is_fully_expanded([]):
                return node
            
            # Select child with highest UCB value
            best_child = max(node.children, key=lambda c: c.get_ucb_value(self.exploration_constant))
            node = best_child
        
        return node
    
    def expand(self, node: MCTSNode, available_actions: List[str]) -> MCTSNode:
        """Expand node with a new child."""
        if not available_actions:
            return node
        
        # Select random unexpanded action
        expanded_actions = [child.action for child in node.children]
        unexpanded_actions = [action for action in available_actions if action not in expanded_actions]
        
        if unexpanded_actions:
            action = np.random.choice(unexpanded_actions)
            # Create dummy state for now (in real implementation, this would be the actual state)
            dummy_state = node.state.copy() if hasattr(node.state, 'copy') else node.state
            child = node.add_child(dummy_state, action)
            return child
        
        return node
    
    def simulate(self, node: MCTSNode) -> float:
        """Simulate random playout from node."""
        # Simple random simulation
        # In a real implementation, this would simulate actual gameplay
        simulation_steps = 10
        total_reward = 0.0
        
        for _ in range(simulation_steps):
            # Random action selection
            reward = np.random.normal(0, 1)  # Dummy reward
            total_reward += reward
        
        return total_reward
    
    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def search(self, root_state, num_simulations: int = 1000) -> Optional[MCTSNode]:
        """Run MCTS search."""
        root = MCTSNode(root_state)
        
        for _ in range(num_simulations):
            node = self.select(root)
            if node.depth < 10:  # Limit depth
                node = self.expand(node, ['action1', 'action2', 'action3'])  # Dummy actions
            value = self.simulate(node)
            self.backpropagate(node, value)
        
        # Return best child
        if root.children:
            return max(root.children, key=lambda c: c.visits)
        return None 