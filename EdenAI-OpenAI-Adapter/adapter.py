import os
from typing import Dict, Any, List, Optional, Sequence, Mapping
import requests
import json

# Import the actual OpenAI SDK
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

class EdenAIClient(OpenAI):
    """
    A client for Eden AI that extends the OpenAI client to provide compatibility
    while routing requests to Eden AI instead.
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: str = "https://api.edenai.run/v2",
        timeout: Optional[float] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        # Initialize the OpenAI client with the given parameters
        super().__init__(
            api_key=api_key, 
            organization=organization,
            base_url=base_url,  # Use Eden AI base URL
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        # Store the API key for use in requests
        self.eden_api_key = api_key or os.environ.get("EDEN_AI_API_KEY")
        if not self.eden_api_key:
            raise ValueError("API key must be provided or set as EDEN_AI_API_KEY environment variable")
        
        # Define Eden AI specific endpoints
        self.eden_chat_endpoint = f"{base_url}/llm/chat"
        
        # Override the chat completions function to use Eden AI
        self.chat = EdenAIChatCompletions(self)

class EdenAIChatCompletions:
    """
    Chat completions implementation for Eden AI that mimics the OpenAI SDK interface.
    """
    
    def __init__(self, client: EdenAIClient):
        self.client = client
    
    def create(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,  # Eden AI doesn't support streaming, but added for compatibility
        temperature: Optional[float] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        tool_definitions: Optional[Any] = None,
        tools: Optional[Any] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs  # Handle any other extra parameters
    ) -> ChatCompletion:
        """
        Create a chat completion using Eden AI.
        
        Args match the OpenAI SDK's create method for compatibility.
        """
        # If streaming is requested, warn that it's not supported
        if stream:
            print("Warning: streaming is not supported by Eden AI and will be ignored")
        
        # Convert OpenAI-style messages to Eden AI format
        eden_messages = self._convert_messages_to_eden_format(messages)
        
        # Prepare the Eden AI request payload
        payload = {
            "model": model,
            "messages": eden_messages
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if n is not None:
            payload["n"] = n
        if stop is not None:
            payload["stop"] = stop
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        
        # Add any extra body parameters
        if extra_body:
            payload.update(extra_body)
        
        # Set up headers for Eden AI
        headers = {
            "Authorization": f"Bearer {self.client.eden_api_key}",
            "Content-Type": "application/json"
        }
        
        # Add any extra headers
        if extra_headers:
            headers.update(extra_headers)
        
        # Make the request to Eden AI
        response = requests.post(
            self.client.eden_chat_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Eden AI API Error: {response.text}")
        
        # Get the response
        eden_response = response.json()
        
        # Process the response based on its format
        processed_response = self._process_response(eden_response, model)
        
        # Return the response as a ChatCompletion object
        return processed_response
    
    def _convert_messages_to_eden_format(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Eden AI format.
        """
        eden_messages = []
        
        for msg in messages:
            # Convert message to dict if it's not already
            if not isinstance(msg, dict):
                # Handle SDK objects by converting to dict
                if hasattr(msg, "model_dump"):
                    msg = msg.model_dump()
                else:
                    # Fallback for other object types
                    msg = {"role": msg.role, "content": msg.content}
            
            # Handle system messages (Eden AI doesn't have system messages, convert to user)
            if msg.get("role") == "system":
                eden_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.get("content", "")}]
                })
                continue
            
            # For regular content (string)
            content = msg.get("content", "")
            if isinstance(content, str):
                eden_messages.append({
                    "role": msg.get("role", "user"),
                    "content": [{"type": "text", "text": content}]
                })
            # For array content (multimodal)
            elif isinstance(content, list):
                content_list = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_list.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            content_list.append({
                                "type": "image_url",
                                "image_url": item.get("image_url", {})
                            })
                
                eden_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content_list
                })
            else:
                eden_messages.append(msg)  # Already in correct format
        
        return eden_messages
    
    def _process_response(self, eden_response: Dict[str, Any], model: str) -> ChatCompletion:
        """
        Process the Eden AI response, handling different response formats.
        """
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
            else:
                # Try to find any valid provider response
                for key in eden_response.keys():
                    if key not in ["status", "error"] and isinstance(eden_response[key], dict):
                        provider_response = eden_response[key]
                        break
                else:
                    raise Exception(f"No valid provider response found in Eden AI response: {eden_response}")
            
            # Convert to OpenAI format
            return self._convert_to_chat_completion(provider_response, model)
        
        # If we reach here, it's likely the new format where Eden AI returns OpenAI format directly
        return self._convert_to_chat_completion(eden_response, model)
    
    def _convert_to_chat_completion(self, response: Dict[str, Any], model: str) -> ChatCompletion:
        """
        Convert the response to a ChatCompletion object.
        """
        # Extract choices
        choices = response.get("choices", [])
        
        # Create Choice objects
        choice_objects = []
        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            choice_objects.append(
                Choice(
                    index=i,
                    message=ChatCompletionMessage(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                        function_call=message.get("function_call"),
                        tool_calls=message.get("tool_calls")
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                    logprobs=None
                )
            )
        
        # Extract usage
        usage_data = response.get("usage", {})
        usage = CompletionUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        # Create and return ChatCompletion object
        return ChatCompletion(
            id=response.get("id", "eden-ai-response"),
            choices=choice_objects,
            created=response.get("created", 0),
            model=model,
            object=response.get("object", "chat.completion"),
            usage=usage,
            system_fingerprint=response.get("system_fingerprint", None)
        )

# Function to create a client that mimics OpenAI's client creation pattern
def EdenAI(api_key: Optional[str] = None, **kwargs) -> EdenAIClient:
    """
    Create an Eden AI client using the OpenAI SDK interface.
    
    Args:
        api_key: Eden AI API key (defaults to EDEN_AI_API_KEY env var)
        **kwargs: Additional arguments to pass to the client
        
    Returns:
        EdenAIClient: Client with OpenAI SDK interface
    """
    return EdenAIClient(api_key=api_key, **kwargs)