"""
LLM-based subgoal generation for GameMind.
"""

import os
from typing import List, Dict, Any, Optional


class SubGoalGenerator:
    """Generate subgoals using LLM for hierarchical planning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the subgoal generator."""
        self.config = config
        self.llm_config = config.get('llm', {})
        self.enabled = self.llm_config.get('enabled', True)
        
        # LLM configuration
        self.model_name = self.llm_config.get('model_name', 'llama-2-7b-chat')
        self.max_tokens = self.llm_config.get('max_tokens', 128)
        self.temperature = self.llm_config.get('temperature', 0.7)
        
        # Prompt templates
        self.prompts = self.llm_config.get('prompts', {})
        
        # Initialize LLM client if available
        self.llm_client = None
        if self.enabled:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client."""
        try:
            from .llama_client import LlamaClient
            self.llm_client = LlamaClient(self.llm_config)
        except ImportError:
            print("LLM client not available, using fallback subgoal generation")
            self.enabled = False
    
    def generate_subgoals(self, goal: str, current_state=None) -> List[str]:
        """Generate subgoals for the given goal."""
        if not self.enabled or not self.llm_client:
            return self._generate_fallback_subgoals(goal)
        
        try:
            # Create prompt for subgoal generation
            prompt = self._create_subgoal_prompt(goal, current_state)
            
            # Generate response using LLM
            response = self.llm_client.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            
            # Parse subgoals from response
            subgoals = self._parse_subgoals(response)
            
            return subgoals[:5]  # Limit to 5 subgoals
            
        except Exception as e:
            print(f"LLM subgoal generation failed: {e}")
            return self._generate_fallback_subgoals(goal)
    
    def _create_subgoal_prompt(self, goal: str, current_state=None) -> str:
        """Create prompt for subgoal generation."""
        base_prompt = self.prompts.get('subgoal_generation', 
                                     "Generate 3-5 specific subgoals for: {goal}")
        
        prompt = base_prompt.format(goal=goal)
        
        if current_state is not None:
            state_info = self._format_state_info(current_state)
            prompt += f"\n\nCurrent state: {state_info}"
        
        return prompt
    
    def _format_state_info(self, state) -> str:
        """Format current state information."""
        if isinstance(state, dict):
            return str(state)
        elif hasattr(state, 'shape'):
            return f"Observation shape: {state.shape}"
        else:
            return str(state)
    
    def _parse_subgoals(self, response: str) -> List[str]:
        """Parse subgoals from LLM response."""
        if not response:
            return []
        
        # Simple parsing - look for numbered or bulleted items
        lines = response.strip().split('\n')
        subgoals = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering/bullets
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*', '•')):
                line = line[2:].strip() if line[1] == '.' else line[1:].strip()
            
            if line and len(line) > 3:
                subgoals.append(line)
        
        return subgoals
    
    def _generate_fallback_subgoals(self, goal: str) -> List[str]:
        """Generate fallback subgoals without LLM."""
        fallback_subgoals = {
            'survive': ['find food', 'find shelter', 'avoid enemies', 'maintain health'],
            'collect wood': ['find trees', 'chop wood', 'gather resources', 'return to base'],
            'make wood_pickaxe': ['collect wood', 'find workbench', 'craft pickaxe', 'test tool'],
            'place furnace': ['collect stone', 'find location', 'place building', 'verify placement'],
            'defeat zombie': ['find weapon', 'approach enemy', 'attack', 'retreat if needed'],
            'explore': ['move around', 'map area', 'find resources', 'avoid danger']
        }
        
        # Find matching goal
        for key, subgoals in fallback_subgoals.items():
            if key.lower() in goal.lower():
                return subgoals
        
        # Default subgoals
        return ['explore environment', 'gather resources', 'avoid danger', 'complete objective']
    
    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze a task to understand requirements."""
        if not self.enabled or not self.llm_client:
            return self._analyze_task_fallback(task)
        
        try:
            prompt = self.prompts.get('task_analysis', 
                                    "Analyze the current task: {task}. What are the key steps needed?")
            prompt = prompt.format(task=task)
            
            response = self.llm_client.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            
            return {
                'task': task,
                'analysis': response,
                'complexity': self._estimate_complexity(response),
                'estimated_steps': self._estimate_steps(response)
            }
            
        except Exception as e:
            print(f"Task analysis failed: {e}")
            return self._analyze_task_fallback(task)
    
    def _analyze_task_fallback(self, task: str) -> Dict[str, Any]:
        """Fallback task analysis."""
        return {
            'task': task,
            'analysis': 'Basic task analysis',
            'complexity': 'medium',
            'estimated_steps': 3
        }
    
    def _estimate_complexity(self, analysis: str) -> str:
        """Estimate task complexity from analysis."""
        if not analysis:
            return 'unknown'
        
        words = analysis.lower().split()
        if len(words) < 10:
            return 'simple'
        elif len(words) < 20:
            return 'medium'
        else:
            return 'complex'
    
    def _estimate_steps(self, analysis: str) -> int:
        """Estimate number of steps from analysis."""
        if not analysis:
            return 3
        
        # Count action words
        action_words = ['collect', 'craft', 'place', 'defeat', 'find', 'move', 'use']
        count = sum(1 for word in action_words if word in analysis.lower())
        
        return max(2, min(count, 6)) 