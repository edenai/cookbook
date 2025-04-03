from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
import requests
import json

# Import our Eden AI adapter
from adapter import EdenAI

# Load environment variables
load_dotenv()

app = FastAPI(title="Eden AI OpenAI SDK API", 
              description="FastAPI application to use Eden AI with OpenAI SDK interface")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for API key
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    print("Warning: EDEN_AI_API_KEY not found in environment variables")

# Dependency to get client
def get_client():
    return EdenAI(api_key=EDEN_AI_API_KEY)

# Models for API requests
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str = "openai/gpt-4o"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

# Text and image chat endpoint
@app.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
    client = Depends(get_client)
):
    """
    Process a chat completion request using Eden AI with OpenAI SDK interface
    """
    try:
        # Convert Pydantic models to dictionaries
        messages = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Extract parameters that are not None
        params = {}
        for key, value in request.dict().items():
            if key != "messages" and value is not None:
                params[key] = value
        
        # Call Eden AI using OpenAI SDK interface
        response = client.chat.create(
            messages=messages,
            **params
        )
        
        # Convert the response to a dictionary for JSON serialization
        response_dict = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        if hasattr(response, 'system_fingerprint') and response.system_fingerprint:
            response_dict["system_fingerprint"] = response.system_fingerprint
        
        # Return the response as JSON
        return JSONResponse(content=response_dict)
    
    except Exception as e:
        error_detail = str(e)
        print(f"Error in chat_completions: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

# Raw Eden AI endpoint for direct testing
@app.post("/eden/raw")
async def eden_raw(
    request: Dict[str, Any] = Body(...)
):
    """
    Send a raw request to Eden AI for testing
    """
    try:
        # Get API key
        headers = {"Authorization": f"Bearer {EDEN_AI_API_KEY}"}
        
        # Make direct request to Eden AI
        response = requests.post("https://api.edenai.run/v2/llm/chat", 
                                json=request, 
                                headers=headers)
        
        # Return the raw response
        return JSONResponse(content=response.json())
    
    except Exception as e:
        error_detail = str(e)
        print(f"Error in eden_raw: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)