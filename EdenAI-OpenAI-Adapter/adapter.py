import os
from typing import Dict, Any, List, Optional, Union
import requests
import json

class EdenAIOpenAIAdapter:
    """
    Adapter class that allows using Eden AI with the OpenAI SDK interface.
    This provides a compatibility layer that maps OpenAI SDK calls to Eden AI API calls.
    """
    
    def __init__(self, api_key: str, base_url: str = None):
        """
        Initialize the adapter with Eden AI API key.
        
        Args:
            api_key: Eden AI API key
            base_url: Ignored, included for compatibility with OpenAI client
        """
        self.api_key = api_key
        self.eden_api_url = "https://api.edenai.run/v2/llm/chat"
        self.chat = self.ChatCompletions(self)
    
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self  # For compatibility with client.chat.completions.create()
        
        def create(self, 
                  model: str, 
                  messages: List[Dict[str, Any]], 
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  **kwargs) -> Dict[str, Any]:
            """
            Create a chat completion using Eden AI.
            Maps OpenAI SDK interface to Eden AI API.
            
            Args:
                model: Model name to use (should be in "provider/model" format for Eden AI)
                messages: List of message dictionaries with "role" and "content"
                temperature: Sampling temperature
                max_tokens: Maximum number of tokens to generate
                **kwargs: Additional arguments (ignored for compatibility)
                
            Returns:
                Dict: OpenAI-compatible response object
            """
            # Prepare Eden AI request
            # Convert messages to Eden AI format if needed
            eden_messages = self._convert_messages_to_eden_format(messages)
            
            # Prepare the Eden AI payload
            payload = {
                "model": model,
                "messages": eden_messages
            }
            
            if temperature is not None:
                payload["temperature"] = temperature
                
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            
            # Make the request to Eden AI
            headers = {"Authorization": f"Bearer {self.parent.api_key}"}
            response = requests.post(self.parent.eden_api_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Eden AI API Error: {response.text}")
            
            # Get the raw response
            eden_response = response.json()
            
            # NEW: Handle both response formats
            # 1. If the response has provider-specific keys (older format)
            # 2. If the response is directly in OpenAI format (newer format)
            
            # Check if this is the older format with provider-specific keys
            if 'openai' in eden_response or 'anthropic' in eden_response or 'google' in eden_response:
                # Extract provider name from model string (e.g., "openai/gpt-4o" -> "openai")
                if '/' in model:
                    provider = model.split('/')[0]
                else:
                    provider = "openai"  # Default to openai if no provider specified
                
                # Get provider-specific response
                if provider in eden_response:
                    provider_response = eden_response[provider]
                    return provider_response
                else:
                    # Try to find any valid provider response
                    for key in eden_response.keys():
                        if key not in ["status", "error"] and isinstance(eden_response[key], dict):
                            return eden_response[key]
            
            # If we reach here, it's likely the new format where Eden AI returns OpenAI format directly
            return eden_response
        
        def _convert_messages_to_eden_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Convert OpenAI-style messages to Eden AI format if needed.
            
            Args:
                messages: OpenAI-style messages
                
            Returns:
                List of Eden AI compatible messages
            """
            eden_messages = []
            
            for msg in messages:
                # Handle system messages (Eden AI doesn't have system messages, convert to user)
                if msg["role"] == "system":
                    eden_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                    continue
                
                # For regular content (string)
                if isinstance(msg["content"], str):
                    eden_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                # For array content (multimodal)
                elif isinstance(msg["content"], list):
                    content_list = []
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            content_list.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image_url":
                            content_list.append({
                                "type": "image_url",
                                "image_url": item["image_url"]
                            })
                    
                    eden_messages.append({
                        "role": msg["role"],
                        "content": content_list
                    })
                else:
                    eden_messages.append(msg)  # Already in correct format
            
            return eden_messages

# Create a function to initialize the client, making it similar to OpenAI client
def EdenAI(api_key: str = None, base_url: str = None):
    """
    Create an Eden AI client with OpenAI SDK interface.
    
    Args:
        api_key: Eden AI API key (defaults to EDEN_AI_API_KEY env var)
        base_url: Ignored, included for compatibility with OpenAI client
        
    Returns:
        EdenAIOpenAIAdapter: Client with OpenAI SDK interface
    """
    if api_key is None:
        api_key = os.environ.get("EDEN_AI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set as EDEN_AI_API_KEY environment variable")
    
    return EdenAIOpenAIAdapter(api_key=api_key)