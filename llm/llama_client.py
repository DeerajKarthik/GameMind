"""
Llama client for GameMind LLM integration.
"""

import os
import requests
from typing import Dict, Any, Optional


class LlamaClient:
    """Client for interacting with Llama models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Llama client."""
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'llama2')
        self.api_key = config.get('api_key', None)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Llama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Connected to Llama server at {self.base_url}")
            else:
                print(f"Warning: Llama server returned status {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Llama server: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate text using Llama model."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Llama API error: {response.status_code}")
                return self._fallback_generation(prompt)
                
        except Exception as e:
            print(f"Llama generation failed: {e}")
            return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback text generation when Llama is unavailable."""
        # Simple rule-based fallback
        if 'collect' in prompt.lower():
            return "1. Find resource location\n2. Approach resource\n3. Use appropriate tool\n4. Gather resource"
        elif 'craft' in prompt.lower():
            return "1. Gather required materials\n2. Find crafting station\n3. Select recipe\n4. Craft item"
        elif 'defeat' in prompt.lower():
            return "1. Find enemy\n2. Equip weapon\n3. Approach carefully\n4. Attack and retreat"
        else:
            return "1. Explore environment\n2. Gather resources\n3. Complete objective\n4. Return to base"
    
    def chat(self, messages: list, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """Chat with Llama model using message history."""
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            return self.generate(prompt, max_tokens, temperature)
        except Exception as e:
            print(f"Chat failed: {e}")
            return "I'm sorry, I'm having trouble processing your request."
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert message history to prompt format."""
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def get_models(self) -> list:
        """Get available models from Llama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []
    
    def is_available(self) -> bool:
        """Check if Llama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False 